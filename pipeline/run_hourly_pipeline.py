#!/usr/bin/env python3
"""
run_hourly_pipeline.py — Alibaba Microservices Trace v2022 Preprocessing Pipeline

Processes MSMetrics, MSRTMCR, and NodeMetrics **one hour at a time** with:

  • Transactional safety   — atomic commit via _SUCCESS marker; final directory
                             NEVER contains partial results for any hour.
  • Resumability           — completed hours are recorded in a JSONL manifest
                             + _SUCCESS marker; all four interruption scenarios
                             are detected and handled on the next run.
  • Stale-file guard       — refuses to proceed if global data/ dirs contain
                             archives from a previous run (unless
                             --allow-global-data-fallback is set).
  • Archive-count check    — verifies the exact expected number of archives
                             per source before any processing begins; fails
                             the hour if any file is missing.
  • No silent CSV skips    — parse errors in required sources are fatal
                             and leave raw archives in place for debugging.
  • Hard validation        — duplicate rows, out-of-window timestamps,
                             negative utilisation values, and empty outputs
                             all abort the commit.
  • Disk guardrails        — hard-abort at 25 GB free (default), warn at
                             40 GB; both are CLI-configurable.
  • Memory guardrails      — hard-abort when RAM % exceeds threshold
                             (requires psutil; gracefully skipped otherwise).
  • Sequential extraction  — one archive at a time; extracted CSV is deleted
                             immediately after aggregation.
  • NodeMetrics cache      — a NodeMetrics archive covers 12 hours; caching
                             it in cache/NodeMetrics/ avoids re-downloading
                             it for hours 1-11, 13-23, etc.
  • Safe tar extraction    — absolute paths and path-traversal sequences
                             (../) are rejected before extraction.

CallGraph is intentionally excluded — too large, not needed for autoscaling.

NOTE ON UNITS
  All utilisation (cpu_utilization, memory_utilization) and call-rate (MCR)
  values are stored exactly as they appear in the Alibaba trace, i.e. they
  are Max-min normalised by Alibaba.  Do NOT interpret them as real CPU
  cores or real call counts per second.

DIRECTORY LAYOUT
  <project_root>/
    pipeline/logs/pipeline.log
    cache/NodeMetrics/           ← reusable 12-hour NodeMetrics archives
    work/hour_XXXXXX/
      raw/                       ← downloaded .tar.gz archives
      extracted/                 ← temporary CSVs (deleted per archive)
      logs/hour.log
    processed/
      final/
        _committed/hour_XXXXXX   ← _SUCCESS marker (written last in commit)
        service_resource/hour_id=XXXXXX/part.parquet
        service_rtmcr/hour_id=XXXXXX/part.parquet
        node_metrics/hour_id=XXXXXX/part.parquet
        joined_service_features/hour_id=XXXXXX/part.parquet
      staging/hour_XXXXXX/       ← written before commit; removed after
      manifests/
        completed_hours.jsonl
        failed_hours.jsonl
        pipeline_state.json

SAFE FIRST TEST
  python pipeline/run_hourly_pipeline.py \\
      --start 0d0 --end 0d1 \\
      --project-root . \\
      --cleanup-on-success true \\
      --disk-abort-gb 25 \\
      --disk-warn-gb 40

MULTI-HOUR RUN
  python pipeline/run_hourly_pipeline.py \\
      --start 0d0 --end 7d0 \\
      --project-root . \\
      --cleanup-on-success true \\
      --disk-abort-gb 30 \\
      --disk-warn-gb 50 \\
      --parquet-compression zstd

FLAGS
  --force                     Reprocess hours already in the completed manifest.
  --keep-raw                  Keep raw archives after success.
  --keep-extracted            Keep extracted CSVs after success.
  --skip-download             Use files already in work/hour_*/raw/.
  --sources A,B,C             MSMetrics, MSRTMCR, NodeMetrics (default: all).
  --parquet-compression X     zstd (default) | snappy | gzip | none.
  --disk-abort-gb N           Hard-abort if free disk < N GB (default 25).
  --disk-warn-gb  N           Warn if free disk < N GB (default 40).
  --mem-abort-pct N           Hard-abort if RAM > N%% (default 90).
  --allow-global-data-fallback  Allow moving archives from data/ if DATA_DIR
                              was not honoured by fetchData.sh.
  --no-nm-cache               Disable the NodeMetrics archive cache.
  --verbose                   DEBUG-level logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import polars as pl
    _POLARS = True
except ImportError:
    _POLARS = False

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

if not _POLARS and not _PANDAS:
    sys.exit("ERROR: Neither polars nor pandas is installed. "
             "Run: pip install pandas pyarrow")

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

VERSION       = "1.2.0"
CHUNK_ROWS    = 500_000       # rows per pandas read_csv chunk
MS_PER_HOUR   = 3_600_000     # trace timestamps are in milliseconds

# Minutes-per-file for each source (determines expected archive counts).
# MSMetrics:   30 min/file  → 2  files per hour
# MSRTMCR:      3 min/file  → 20 files per hour
# NodeMetrics: 720 min/file → 1  file  per 12 hours
ARCHIVE_RATIOS: dict[str, int] = {
    "msmetrics":   30,
    "msrtmcr":      3,
    "nodemetrics": 720,
}

# Maps source key → (subdir name, file prefix)
SOURCE_META: dict[str, tuple[str, str]] = {
    "msmetrics":   ("MSMetrics",   "MSMetrics"),
    "msrtmcr":     ("MSRTMCR",     "MSRTMCR"),
    "nodemetrics": ("NodeMetrics", "NodeMetrics"),
}

TABLES          = ["service_resource", "service_rtmcr",
                   "node_metrics", "joined_service_features"]
REQUIRED_TABLES = ["service_resource", "joined_service_features"]

MCR_COLS = [
    "providerrpc_mcr", "consumerrpc_mcr",
    "writemc_mcr",     "readmc_mcr",
    "writedb_mcr",     "readdb_mcr",
    "consumermq_mcr",  "providermq_mcr",
    "http_mcr",
]
RT_COLS = [
    "providerrpc_rt", "consumerrpc_rt",
    "writemc_rt",     "readmc_rt",
    "writedb_rt",     "readdb_rt",
    "consumermq_rt",  "providermq_rt",
    "http_rt",
]

def setup_logging(log_file: Optional[Path], verbose: bool) -> None:
    level   = logging.DEBUG if verbose else logging.INFO
    fmt     = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file)))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)


def add_file_handler(log_file: Path) -> logging.FileHandler:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(str(log_file))
    h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(h)
    return h


def remove_handler(h: logging.Handler) -> None:
    logging.getLogger().removeHandler(h)
    h.close()


def parse_date(s: str) -> tuple[int, int]:
    """Parse '3d14' → (day=3, hour=14)."""
    m = re.fullmatch(r"(\d+)d(\d+)", s.strip())
    if not m:
        raise ValueError(f"Invalid date {s!r}. Expected NdM e.g. 0d0 or 3d14.")
    return int(m.group(1)), int(m.group(2))


def to_abs(day: int, hour: int) -> int:
    return day * 24 + hour


def from_abs(h: int) -> tuple[int, int]:
    return h // 24, h % 24


def hid(h: int) -> str:
    """Zero-padded 6-digit hour ID string, e.g. 000042."""
    return f"{h:06d}"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(str(path)).free / 1e9


def mem_pct() -> float:
    return psutil.virtual_memory().percent if _PSUTIL else 0.0


def check_disk(path: Path, label: str, abort_gb: float, warn_gb: float) -> None:
    gb = free_gb(path)
    if gb < abort_gb:
        raise RuntimeError(
            f"DISK GUARDRAIL: {gb:.2f} GB free at {path} — need > {abort_gb} GB "
            f"before {label}. Free space and retry."
        )
    if gb < warn_gb:
        logging.warning(
            f"LOW DISK ({label}): {gb:.2f} GB free "
            f"(warn threshold {warn_gb} GB, abort threshold {abort_gb} GB)."
        )


def check_mem(label: str, abort_pct: float) -> None:
    pct = mem_pct()
    if pct and pct >= abort_pct:
        raise RuntimeError(
            f"MEMORY GUARDRAIL: {pct:.1f}% RAM used before {label}. "
            f"Threshold is {abort_pct}%. Free memory and retry."
        )
    if pct and pct >= abort_pct * 0.85:
        logging.warning(f"HIGH MEM ({label}): {pct:.1f}% RAM used.")


def log_disk(root: Path, label: str) -> None:
    st    = shutil.disk_usage(str(root))
    total = st.total / 1e9
    free  = st.free  / 1e9
    logging.info(f"  Disk [{label}]: {total-free:.2f}/{total:.2f} GB used, "
                 f"{free:.2f} GB free")


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


class Paths:
    def __init__(self, root: Path):
        self.root      = root.resolve()
        self.pipeline  = self.root / "pipeline"
        self.pipe_logs = self.pipeline / "logs"
        self.data      = self.root / "data"
        self.work      = self.root / "work"
        self.cache_dir = self.root / "cache"
        self.processed = self.root / "processed"
        self.final     = self.processed / "final"
        self.staging   = self.processed / "staging"
        self.manifests = self.processed / "manifests"

    def hour_work(self, h: int) -> Path: return self.work  / f"hour_{hid(h)}"
    def hour_raw (self, h: int) -> Path: return self.hour_work(h) / "raw"
    def hour_ext (self, h: int) -> Path: return self.hour_work(h) / "extracted"
    def hour_stg (self, h: int) -> Path: return self.staging / f"hour_{hid(h)}"
    def hour_logs(self, h: int) -> Path: return self.hour_work(h) / "logs"

    def final_part  (self, table: str, h: int) -> Path:
        return self.final / table / f"hour_id={hid(h)}"
    def staging_part(self, table: str, h: int) -> Path:
        return self.hour_stg(h) / table

    def success_marker(self, h: int) -> Path:
        """Written last during commit; its presence means all tables are in final/."""
        return self.final / "_committed" / f"hour_{hid(h)}"

    def nm_cache(self, idx: int) -> Path:
        return self.cache_dir / "NodeMetrics" / f"NodeMetrics_{idx}.tar.gz"

    @property
    def completed(self) -> Path: return self.manifests / "completed_hours.jsonl"
    @property
    def failed   (self) -> Path: return self.manifests / "failed_hours.jsonl"
    @property
    def state    (self) -> Path: return self.manifests / "pipeline_state.json"

    def init(self, h: int) -> None:
        for d in [
            self.pipe_logs,
            self.manifests,
            *[self.final / t for t in TABLES],
            self.final / "_committed",
            self.cache_dir / "NodeMetrics",
            self.hour_raw(h),
            self.hour_ext(h),
            self.hour_stg(h),
            self.hour_logs(h),
        ]:
            d.mkdir(parents=True, exist_ok=True)


def expected_archive_indices(source_key: str, hour_id: int) -> list[int]:
    """
    Return the list of file indices that cover the given absolute hour.

    Logic mirrors fetchData.sh exactly:
      start_idx = (hour_id * 60) // ratio
      end_idx   = ((hour_id+1) * 60) // ratio - 1
      if end_minute % ratio != 0: end_idx += 1
    """
    ratio     = ARCHIVE_RATIOS[source_key]
    start_min = hour_id * 60
    end_min   = start_min + 60
    start_idx = start_min // ratio
    end_idx   = end_min   // ratio - 1
    if end_min % ratio != 0:
        end_idx += 1
    return list(range(start_idx, end_idx + 1))


def expected_archive_names(source_key: str, hour_id: int) -> list[str]:
    _, prefix = SOURCE_META[source_key]
    return [f"{prefix}_{i}.tar.gz"
            for i in expected_archive_indices(source_key, hour_id)]


def validate_downloads(raw_dir: Path, sources: set[str], hour_id: int) -> None:
    """
    Raise RuntimeError if any expected archive is absent or zero-length.
    Called immediately after download / after cache population.
    """
    errors: list[str] = []
    for src in sorted(sources):
        subdir, _ = SOURCE_META[src]
        src_dir   = raw_dir / subdir
        for name in expected_archive_names(src, hour_id):
            p = src_dir / name
            if not p.exists():
                errors.append(f"  MISSING:  {src}/{name}")
            elif p.stat().st_size == 0:
                errors.append(f"  EMPTY:    {src}/{name}  (0 bytes)")
    if errors:
        raise RuntimeError(
            f"Archive validation failed for hour {hour_id} "
            f"({len(errors)} issue(s)):\n" + "\n".join(errors)
        )


def load_completed(paths: Paths) -> set[int]:
    if not paths.completed.exists():
        return set()
    out: set[int] = set()
    with paths.completed.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.add(json.loads(line)["hour_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return out


def _parquets_valid(paths: Paths, h: int) -> bool:
    """Return True iff all REQUIRED_TABLES have a non-empty, readable parquet."""
    for table in REQUIRED_TABLES:
        pq = paths.final_part(table, h) / "part.parquet"
        if not pq.exists() or pq.stat().st_size == 0:
            return False
        try:
            _quick_read(pq)
        except Exception:
            return False
    return True


def is_completed(paths: Paths, h: int) -> bool:
    """
    Four-state completion check with automatic recovery:

    State A — manifest + marker + valid parquets  → completed ✓
    State B — manifest only (old pipeline)         → write marker, return True
    State C — marker only (crash between commit
              and manifest write)                   → write manifest, return True
    State D / E — neither                          → return False
                  (partial final partitions are
                   cleaned up by cleanup_partial_commit())
    """
    in_manifest = h in load_completed(paths)
    has_marker  = paths.success_marker(h).exists()

    if in_manifest and has_marker:
        # State A: manifest + marker. Still verify the parquets: final/ may have
        # been deleted or corrupted after a successful commit.
        if _parquets_valid(paths, h):
            return True
        # Parquets missing/unreadable: remove the stale marker and reprocess.
        logging.warning(
            f"Hour {h}: manifest + _SUCCESS marker exist but required parquets "
            "are missing or unreadable — removing stale marker and reprocessing."
        )
        paths.success_marker(h).unlink(missing_ok=True)
        return False

    if in_manifest and not has_marker:
        # State B: old pipeline run without markers, or marker was deleted
        if _parquets_valid(paths, h):
            paths.success_marker(h).parent.mkdir(parents=True, exist_ok=True)
            paths.success_marker(h).touch()
            logging.debug(f"Hour {h}: wrote missing _SUCCESS marker (backward compat).")
            return True
        logging.warning(
            f"Hour {h} is in the manifest but required parquets are "
            "missing or unreadable — will reprocess."
        )
        return False

    if has_marker and not in_manifest:
        # State C: crash between commit_hour() and write_completed()
        if _parquets_valid(paths, h):
            logging.info(
                f"Hour {h}: _SUCCESS marker exists without manifest entry — "
                "repairing manifest."
            )
            write_completed(paths, h, {"recovered": True, "source": "success_marker_repair"})
            return True
        # Marker present but parquets missing/bad — remove marker, retry
        logging.warning(
            f"Hour {h}: _SUCCESS marker exists but parquets are invalid — "
            "removing marker, will reprocess."
        )
        paths.success_marker(h).unlink(missing_ok=True)
        return False

    # State D/E: neither marker nor manifest entry
    return False


def write_completed(paths: Paths, h: int, meta: dict) -> None:
    rec = {"hour_id": h,
           "completed_at": datetime.now(timezone.utc).isoformat(),
           **meta}
    with paths.completed.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def write_failed(paths: Paths, h: int, reason: str) -> None:
    rec = {"hour_id": h,
           "failed_at": datetime.now(timezone.utc).isoformat(),
           "reason": str(reason)[:4000]}
    with paths.failed.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def cleanup_partial_commit(paths: Paths, h: int) -> None:
    """
    Called at the start of run_hour (Scenario D recovery):
    If any final partitions exist without a _SUCCESS marker, the previous
    commit was interrupted mid-flight.  Remove the partial tables so that
    this run starts clean.
    """
    if paths.success_marker(h).exists():
        return  # complete commit — do not touch

    any_partial = any(paths.final_part(t, h).exists() for t in TABLES)
    if not any_partial:
        return

    logging.warning(
        f"  Hour {h}: partial final partitions found without _SUCCESS marker "
        "(interrupted commit from previous run) — removing for clean retry."
    )
    for t in TABLES:
        p = paths.final_part(t, h)
        if p.exists():
            shutil.rmtree(str(p))
            logging.info(f"    Removed partial partition: {t}/hour_id={hid(h)}")


def _quick_read(path: Path) -> int:
    """Read a parquet and return its row count (used in validation)."""
    if _POLARS:
        return len(pl.read_parquet(str(path)))
    return len(pd.read_parquet(str(path)))


def save_parquet(df: "pd.DataFrame", out: Path, compression: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out), compression=(None if compression == "none" else compression),
                  index=False)


def read_csv(path: Path,
             ts_start: Optional[int] = None,
             ts_end:   Optional[int] = None,
             abort_mem_pct: float = 90.0) -> "pd.DataFrame":
    """
    Read a CSV file in CHUNK_ROWS-sized chunks.
    Column names are normalised to lowercase/stripped.
    Optionally filters rows to [ts_start, ts_end) on 'timestamp'.
    Raises on I/O or parse errors — callers must NOT silently catch these.
    """
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(str(path), chunksize=CHUNK_ROWS, low_memory=False):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        if ts_start is not None and "timestamp" in chunk.columns:
            chunk = chunk[
                (chunk["timestamp"] >= ts_start) &
                (chunk["timestamp"] <  ts_end)
            ]
        if not chunk.empty:
            parts.append(chunk)
        check_mem(f"reading {path.name}", abort_mem_pct)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def extract(gz: Path, dest: Path) -> list[Path]:
    """
    Safely extract a .tar.gz into dest.
    Rejects:  absolute member paths
              path traversal sequences (..)
    Extracts: regular files only
    """
    dest.mkdir(parents=True, exist_ok=True)
    dest_real = dest.resolve()
    extracted: list[Path] = []

    with tarfile.open(str(gz), "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            if os.path.isabs(member.name):
                raise RuntimeError(
                    f"UNSAFE ARCHIVE {gz.name}: member has absolute path "
                    f"'{member.name}'. Aborting extraction."
                )

            # Use Path.relative_to() instead of str.startswith() to avoid
            # false negatives from prefix collisions, e.g.:
            #   /tmp/safe-dir  vs  /tmp/safe-dir-evil/../../...
            target = (dest / member.name).resolve()
            try:
                target.relative_to(dest_real)
            except ValueError:
                raise RuntimeError(
                    f"UNSAFE ARCHIVE {gz.name}: member '{member.name}' would "
                    f"escape extraction directory {dest}. Aborting."
                )

            tar.extract(member, str(dest))
            extracted.append(dest / member.name)

    return extracted


def _idx(p: Path) -> int:
    """Numeric suffix from e.g. MSMetrics_7.tar.gz → 7."""
    m = re.search(r"_(\d+)\.tar\.gz$", p.name)
    return int(m.group(1)) if m else 0


def populate_nm_from_cache(paths: Paths, raw_dir: Path,
                            hour_id: int, no_cache: bool) -> set[int]:
    """
    Copy cached NodeMetrics archives into raw_dir/NodeMetrics/ before
    running fetchData.sh, which skips files that already exist.

    Returns the set of indices that were satisfied from cache (so we don't
    re-cache them after download).
    """
    if no_cache:
        return set()

    indices  = expected_archive_indices("nodemetrics", hour_id)
    dest_dir = raw_dir / "NodeMetrics"
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached: set[int] = set()

    for idx in indices:
        src  = paths.nm_cache(idx)
        dst  = dest_dir / f"NodeMetrics_{idx}.tar.gz"
        if src.exists() and src.stat().st_size > 0:
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
                logging.info(f"  NM cache hit: NodeMetrics_{idx}.tar.gz "
                             f"({src.stat().st_size/1e6:.1f} MB) → {dst.parent.name}/")
            else:
                logging.info(f"  NM cache hit (already in raw): NodeMetrics_{idx}.tar.gz")
            cached.add(idx)
        else:
            logging.info(f"  NM cache miss: NodeMetrics_{idx}.tar.gz — will download.")

    return cached


def save_nm_to_cache(paths: Paths, raw_dir: Path,
                     hour_id: int, no_cache: bool,
                     already_cached: set[int]) -> None:
    """
    After a successful download, copy newly-downloaded NodeMetrics archives
    to the cache so that subsequent hours in the same 12-hour block can reuse them.
    """
    if no_cache:
        return

    src_dir = raw_dir / "NodeMetrics"
    for idx in expected_archive_indices("nodemetrics", hour_id):
        if idx in already_cached:
            continue
        src = src_dir / f"NodeMetrics_{idx}.tar.gz"
        if not src.exists() or src.stat().st_size == 0:
            continue
        dst = paths.nm_cache(idx)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        logging.info(f"  NM cached: NodeMetrics_{idx}.tar.gz "
                     f"({src.stat().st_size/1e6:.1f} MB) → cache/NodeMetrics/")


def _agg_ms_metrics(df: "pd.DataFrame") -> "pd.DataFrame":
    """Aggregate raw instance-level MSMetrics to service level per timestamp."""
    grp = df.groupby(["timestamp", "msname"])

    counts = grp.agg(
        replica_count=("msinstanceid", "nunique"),
        node_count   =("nodeid",       "nunique"),
    )
    cpu   = grp["cpu_utilization"].agg(["sum", "mean", "max"])
    cpu.columns = ["cpu_sum", "cpu_mean", "cpu_max"]
    cpu_q = grp["cpu_utilization"].quantile([0.50, 0.95]).unstack(level=-1)
    cpu_q.columns = ["cpu_p50", "cpu_p95"]
    mem   = grp["memory_utilization"].agg(["sum", "mean", "max"])
    mem.columns = ["memory_sum", "memory_mean", "memory_max"]
    mem_q = grp["memory_utilization"].quantile([0.50, 0.95]).unstack(level=-1)
    mem_q.columns = ["memory_p50", "memory_p95"]

    return pd.concat([counts, cpu, cpu_q, mem, mem_q], axis=1).reset_index()


def process_msmetrics(raw_dir: Path, ext_dir: Path,
                      hour_id: int, cfg: argparse.Namespace) -> "pd.DataFrame":
    """
    Process all MSMetrics archives for this hour, one archive at a time.
    Archives cover non-overlapping 30-min windows, so results are concatenated.
    Any CSV parse error is fatal — no silent skipping.
    """
    ts_start = hour_id * MS_PER_HOUR
    ts_end   = ts_start + MS_PER_HOUR
    src      = raw_dir / "MSMetrics"
    expected = expected_archive_names("msmetrics", hour_id)
    files    = [src / name for name in expected]

    # Detect unexpected extra archives (could indicate wrong DATA_DIR or
    # leftover files from a different hour) and fail loudly.
    actual_names = {p.name for p in src.glob("*.tar.gz")} if src.exists() else set()
    unexpected = actual_names - set(expected)
    if unexpected:
        raise RuntimeError(
            f"MSMetrics dir contains unexpected archives for hour {hour_id}: "
            f"{sorted(unexpected)}. Expected only: {expected}. "
            f"This may indicate stale data from a different hour."
        )

    logging.info(f"  MSMetrics: {len(files)} archive(s) "
                 f"(expected: {expected})")
    tmp       = ext_dir / "MSMetrics"
    agg_parts: list[pd.DataFrame] = []
    total_raw_rows      = 0
    total_raw_instances = 0

    for gz in files:
        check_mem(f"MSMetrics {gz.name}", cfg.mem_abort_pct)
        check_disk(raw_dir.parent.parent, f"extract {gz.name}",
                   cfg.disk_abort_gb, cfg.disk_warn_gb)

        csvs = extract(gz, tmp)
        logging.info(f"    {gz.name} ({gz.stat().st_size/1e6:.1f} MB) → "
                     f"{[c.name for c in csvs]}")

        for csv in csvs:
            try:
                df = read_csv(csv, ts_start, ts_end, cfg.mem_abort_pct)
            except Exception as e:
                csv.unlink(missing_ok=True)
                raise RuntimeError(
                    f"MSMetrics: failed to read {csv.name}: {e}"
                ) from e

            if df.empty:
                logging.debug(f"      {csv.name}: no rows in hour window [skip].")
                csv.unlink(missing_ok=True)
                continue

            n_rows      = len(df)
            n_services  = df["msname"].nunique() if "msname" in df.columns else 0
            n_instances = (df["msinstanceid"].nunique()
                           if "msinstanceid" in df.columns else 0)
            total_raw_rows      += n_rows
            total_raw_instances += n_instances
            logging.info(f"      {csv.name}: {n_rows:,} rows, "
                         f"{n_services:,} services, {n_instances:,} instances")

            agg_parts.append(_agg_ms_metrics(df))
            del df
            csv.unlink(missing_ok=True)

    if not agg_parts:
        raise RuntimeError(
            f"MSMetrics: zero rows produced for hour {hour_id}. "
            "Check timestamp alignment or archive contents."
        )

    result = pd.concat(agg_parts, ignore_index=True)
    logging.info(f"  MSMetrics aggregated: {len(result):,} service-level rows, "
                 f"{result['msname'].nunique():,} unique services, "
                 f"{total_raw_instances:,} raw instances read")
    return result


def _agg_msrtmcr(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Aggregate raw instance-level MSRTMCR to service level per timestamp.
    Timestamps do not overlap between archives, so results are concatenated.
    """
    grp    = df.groupby(["timestamp", "msname"])
    pieces: list[pd.DataFrame] = []

    for col in MCR_COLS:
        if col not in df.columns:
            continue
        agg = grp[col].agg(["sum", "mean", "max"])
        agg.columns = [f"{col}_sum", f"{col}_mean", f"{col}_max"]
        pieces.append(agg)

    for col in RT_COLS:
        if col not in df.columns:
            continue
        base = grp[col].agg(["mean", "max"])
        base.columns = [f"{col}_mean", f"{col}_max"]
        p50  = grp[col].quantile(0.50).rename(f"{col}_p50")
        p95  = grp[col].quantile(0.95).rename(f"{col}_p95")
        pieces.extend([base, p50.to_frame(), p95.to_frame()])

    if not pieces:
        return pd.DataFrame()

    result = pd.concat(pieces, axis=1).reset_index()

    # Derived convenience columns
    def _safe_sum(df: "pd.DataFrame", cols: list[str]) -> "pd.Series":
        present = [c for c in cols if c in df.columns]
        return df[present].sum(axis=1) if present else pd.Series(0.0, index=df.index)

    result["total_mcr_sum"]    = _safe_sum(result, [f"{c}_sum" for c in MCR_COLS])
    result["rpc_mcr_sum"]      = _safe_sum(result, ["providerrpc_mcr_sum",
                                                     "consumerrpc_mcr_sum"])
    result["mq_mcr_sum"]       = _safe_sum(result, ["providermq_mcr_sum",
                                                     "consumermq_mcr_sum"])
    result["stateful_mcr_sum"] = _safe_sum(result, ["writemc_mcr_sum", "readmc_mcr_sum",
                                                     "writedb_mcr_sum", "readdb_mcr_sum"])
    return result


def process_msrtmcr(raw_dir: Path, ext_dir: Path,
                    hour_id: int, cfg: argparse.Namespace) -> "pd.DataFrame":
    """
    Process all MSRTMCR archives (20 per hour at 3 min/file), one at a time.
    Any CSV parse error is fatal — no silent skipping.
    """
    ts_start = hour_id * MS_PER_HOUR
    ts_end   = ts_start + MS_PER_HOUR
    src      = raw_dir / "MSRTMCR"
    expected = expected_archive_names("msrtmcr", hour_id)
    files    = [src / name for name in expected]

    actual_names = {p.name for p in src.glob("*.tar.gz")} if src.exists() else set()
    unexpected = actual_names - set(expected)
    if unexpected:
        raise RuntimeError(
            f"MSRTMCR dir contains unexpected archives for hour {hour_id}: "
            f"{sorted(unexpected)}. Expected only: {expected}. "
            f"This may indicate stale data from a different hour."
        )

    logging.info(f"  MSRTMCR: {len(files)} archive(s) "
                 f"(expected: {len(expected)} files, indices "
                 f"{expected_archive_indices('msrtmcr', hour_id)[0]}–"
                 f"{expected_archive_indices('msrtmcr', hour_id)[-1]})")
    tmp       = ext_dir / "MSRTMCR"
    agg_parts: list[pd.DataFrame] = []
    total_raw_rows = 0

    for gz in files:
        check_mem(f"MSRTMCR {gz.name}", cfg.mem_abort_pct)
        check_disk(raw_dir.parent.parent, f"extract {gz.name}",
                   cfg.disk_abort_gb, cfg.disk_warn_gb)

        csvs = extract(gz, tmp)

        for csv in csvs:
            try:
                df = read_csv(csv, ts_start, ts_end, cfg.mem_abort_pct)
            except Exception as e:
                csv.unlink(missing_ok=True)
                raise RuntimeError(
                    f"MSRTMCR: failed to read {csv.name}: {e}"
                ) from e

            if df.empty:
                logging.debug(f"      {csv.name}: no rows in hour window [skip].")
                csv.unlink(missing_ok=True)
                continue

            n_rows     = len(df)
            n_services = df["msname"].nunique() if "msname" in df.columns else 0
            total_raw_rows += n_rows
            logging.info(f"      {csv.name}: {n_rows:,} rows, {n_services:,} services")

            agg_parts.append(_agg_msrtmcr(df))
            del df
            csv.unlink(missing_ok=True)

    if not agg_parts:
        raise RuntimeError(
            f"MSRTMCR: zero rows produced for hour {hour_id}. "
            "Check timestamp alignment or archive contents."
        )

    result = pd.concat(agg_parts, ignore_index=True)
    logging.info(f"  MSRTMCR aggregated: {len(result):,} service-level rows, "
                 f"{result['msname'].nunique():,} unique services "
                 f"(from {total_raw_rows:,} raw rows)")
    return result


def process_nodemetrics(raw_dir: Path, ext_dir: Path,
                        hour_id: int, cfg: argparse.Namespace) -> "pd.DataFrame":
    """
    Process NodeMetrics archive.
    One file covers 12 hours → filter strictly to this hour's window.
    Output: timestamp, nodeid, node_cpu_utilization, node_memory_utilization.
    Any CSV parse error is fatal.
    """
    ts_start = hour_id * MS_PER_HOUR
    ts_end   = ts_start + MS_PER_HOUR
    src      = raw_dir / "NodeMetrics"
    expected = expected_archive_names("nodemetrics", hour_id)
    files    = [src / name for name in expected]

    actual_names = {p.name for p in src.glob("*.tar.gz")} if src.exists() else set()
    unexpected = actual_names - set(expected)
    if unexpected:
        raise RuntimeError(
            f"NodeMetrics dir contains unexpected archives for hour {hour_id}: "
            f"{sorted(unexpected)}. Expected only: {expected}. "
            f"This may indicate stale data from a different hour."
        )

    logging.info(f"  NodeMetrics: {len(files)} archive(s) (expected: {expected})")
    tmp   = ext_dir / "NodeMetrics"
    parts: list[pd.DataFrame] = []

    for gz in files:
        check_mem(f"NodeMetrics {gz.name}", cfg.mem_abort_pct)
        check_disk(raw_dir.parent.parent, f"extract {gz.name}",
                   cfg.disk_abort_gb, cfg.disk_warn_gb)

        csvs = extract(gz, tmp)
        logging.info(f"    {gz.name} ({gz.stat().st_size/1e6:.1f} MB) → "
                     f"{[c.name for c in csvs]}")

        for csv in csvs:
            try:
                df = read_csv(csv, ts_start, ts_end, cfg.mem_abort_pct)
            except Exception as e:
                csv.unlink(missing_ok=True)
                raise RuntimeError(
                    f"NodeMetrics: failed to read {csv.name}: {e}"
                ) from e

            if df.empty:
                logging.debug(f"      {csv.name}: no rows in hour window [skip].")
                csv.unlink(missing_ok=True)
                continue

            n_nodes = df["nodeid"].nunique() if "nodeid" in df.columns else 0
            logging.info(f"      {csv.name}: {len(df):,} rows, {n_nodes:,} nodes")

            rename: dict[str, str] = {}
            if "cpu_utilization"    in df.columns:
                rename["cpu_utilization"]    = "node_cpu_utilization"
            if "memory_utilization" in df.columns:
                rename["memory_utilization"] = "node_memory_utilization"
            if rename:
                df = df.rename(columns=rename)

            keep = [c for c in ["timestamp", "nodeid",
                                 "node_cpu_utilization", "node_memory_utilization"]
                    if c in df.columns]
            parts.append(df[keep].sort_values(["timestamp", "nodeid"]))
            csv.unlink(missing_ok=True)

    if not parts:
        raise RuntimeError(
            f"NodeMetrics: zero rows produced for hour {hour_id}. "
            "Check timestamp alignment or archive contents."
        )

    result = (pd.concat(parts, ignore_index=True)
                .sort_values(["timestamp", "nodeid"])
                .reset_index(drop=True))
    logging.info(f"  NodeMetrics aggregated: {len(result):,} rows, "
                 f"{result['nodeid'].nunique():,} unique nodes")
    return result


_JOINED_BASE = [
    "timestamp", "msname", "replica_count", "node_count",
    "cpu_sum",    "cpu_mean",    "cpu_p95",    "cpu_max",
    "memory_sum", "memory_mean", "memory_p95", "memory_max",
]
_JOINED_RT = [
    "total_mcr_sum", "rpc_mcr_sum", "mq_mcr_sum",
    "stateful_mcr_sum", "http_mcr_sum",
    "providerrpc_rt_mean", "providerrpc_rt_p95",
    "consumerrpc_rt_mean", "consumerrpc_rt_p95",
    "http_rt_mean",        "http_rt_p95",
]


def build_joined(svc_res: "pd.DataFrame",
                 svc_rt:  Optional["pd.DataFrame"]) -> "pd.DataFrame":
    """Left-join service_resource + service_rtmcr on (timestamp, msname)."""
    logging.info("  Building joined_service_features…")

    if svc_rt is not None and not svc_rt.empty:
        joined = pd.merge(svc_res, svc_rt, on=["timestamp", "msname"], how="left")
    else:
        logging.warning("  service_rtmcr is empty — joined table will lack RT/MCR columns.")
        joined = svc_res.copy()

    out_cols = _JOINED_BASE + [c for c in _JOINED_RT if c in joined.columns]
    joined   = joined[[c for c in out_cols if c in joined.columns]]
    logging.info(f"  joined_service_features: {len(joined):,} rows, "
                 f"{joined['msname'].nunique():,} services")
    return joined


def validate_staging(paths: Paths, h: int) -> dict:
    """
    Validate all staged outputs before the atomic commit.
    Raises ValueError on ANY data quality problem.
    Returns a stats dict on success.
    """
    stats: dict = {}
    ts_lo = h * MS_PER_HOUR
    ts_hi = ts_lo + MS_PER_HOUR

    pq = paths.staging_part("service_resource", h) / "part.parquet"
    if not pq.exists():
        raise ValueError("Validation failed: staging/service_resource/part.parquet missing.")

    sr = pd.read_parquet(str(pq))
    if sr.empty:
        raise ValueError("Validation failed: service_resource output is empty.")

    for col in ["timestamp", "msname", "replica_count",
                "cpu_sum", "cpu_mean", "memory_sum", "memory_mean"]:
        if col not in sr.columns:
            raise ValueError(
                f"Validation failed: service_resource missing required column '{col}'.")

    bad_rep = int((sr["replica_count"] <= 0).sum())
    if bad_rep:
        raise ValueError(
            f"Validation failed: {bad_rep} rows in service_resource have "
            "replica_count ≤ 0.")

    bad_cpu = int((sr["cpu_sum"] < 0).sum())
    if bad_cpu:
        raise ValueError(
            f"Validation failed: {bad_cpu} rows in service_resource have negative cpu_sum.")

    bad_mem = int((sr["memory_sum"] < 0).sum())
    if bad_mem:
        raise ValueError(
            f"Validation failed: {bad_mem} rows in service_resource have negative memory_sum.")

    sr_dupes = int(sr.duplicated(subset=["timestamp", "msname"]).sum())
    if sr_dupes:
        raise ValueError(
            f"Validation failed: {sr_dupes} duplicate (timestamp, msname) rows "
            "in service_resource.")

    sr_oor = int(((sr["timestamp"] < ts_lo) | (sr["timestamp"] >= ts_hi)).sum())
    if sr_oor:
        raise ValueError(
            f"Validation failed: {sr_oor} rows in service_resource are outside "
            f"the expected hour window [{ts_lo}, {ts_hi}).")

    stats["service_resource_rows"]     = len(sr)
    stats["service_resource_services"] = int(sr["msname"].nunique())
    del sr

    # service_rtmcr (optional)
    pq = paths.staging_part("service_rtmcr", h) / "part.parquet"
    if pq.exists():
        rt = pd.read_parquet(str(pq))
        stats["service_rtmcr_rows"]     = len(rt)
        stats["service_rtmcr_services"] = int(rt["msname"].nunique()) if not rt.empty else 0
        del rt

    # node_metrics (optional)
    pq = paths.staging_part("node_metrics", h) / "part.parquet"
    if pq.exists():
        nm = pd.read_parquet(str(pq))
        stats["node_metrics_rows"]  = len(nm)
        stats["node_metrics_nodes"] = int(nm["nodeid"].nunique()) if not nm.empty else 0
        del nm

    pq = paths.staging_part("joined_service_features", h) / "part.parquet"
    if not pq.exists():
        raise ValueError(
            "Validation failed: staging/joined_service_features/part.parquet missing.")

    jf = pd.read_parquet(str(pq))
    if jf.empty:
        raise ValueError("Validation failed: joined_service_features output is empty.")

    jf_dupes = int(jf.duplicated(subset=["timestamp", "msname"]).sum())
    if jf_dupes:
        raise ValueError(
            f"Validation failed: {jf_dupes} duplicate (timestamp, msname) rows "
            "in joined_service_features.")

    jf_oor = int(((jf["timestamp"] < ts_lo) | (jf["timestamp"] >= ts_hi)).sum())
    if jf_oor:
        raise ValueError(
            f"Validation failed: {jf_oor} rows in joined_service_features are outside "
            f"the expected hour window [{ts_lo}, {ts_hi}).")

    stats["joined_rows"]     = len(jf)
    stats["joined_services"] = int(jf["msname"].nunique())
    del jf

    for k, v in sorted(stats.items()):
        logging.info(f"  ✓ {k}: {v:,}")

    return stats


def commit_hour(paths: Paths, h: int) -> None:
    """
    Move staged partitions into processed/final/ then write _SUCCESS marker.

    Crash safety:
      • If the process dies while moving tables → no _SUCCESS, no manifest.
        cleanup_partial_commit() will detect and remove partial tables on next run.
      • If the process dies after moving tables but before _SUCCESS → no _SUCCESS.
        Same detection applies.
      • If the process dies after _SUCCESS but before manifest write → State C
        in is_completed(), which self-repairs the manifest on next run.
    """
    for table in TABLES:
        src = paths.staging_part(table, h)
        if not src.exists():
            continue
        dst = paths.final_part(table, h)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(str(dst))
        shutil.move(str(src), str(dst))
        logging.info(f"  Moved → {table}/hour_id={hid(h)}  "
                     f"({dir_size_gb(dst)*1000:.1f} MB)")

    # Write _SUCCESS marker last — its presence is the authoritative signal
    # that the commit is complete.
    marker = paths.success_marker(h)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    logging.info(f"  _SUCCESS marker written: {marker}")


def cleanup_hour(paths: Paths, h: int, keep_raw: bool, keep_extracted: bool) -> None:
    if not keep_raw:
        d = paths.hour_raw(h)
        if d.exists():
            shutil.rmtree(str(d))
            logging.info(f"  Deleted raw/  ({dir_size_gb(d)*0:.0f})")
    if not keep_extracted:
        d = paths.hour_ext(h)
        if d.exists():
            shutil.rmtree(str(d))
            logging.info(f"  Deleted extracted/")
    d = paths.hour_stg(h)
    if d.exists():
        shutil.rmtree(str(d))
        logging.info(f"  Deleted staging/")


def download_hour(paths: Paths, h: int, sources: set[str],
                  cfg: argparse.Namespace) -> None:
    """
    Download archives for hour h into work/hour_XXXXXX/raw/.

    Steps:
      1. Pre-populate NodeMetrics from cache (avoids re-downloading 12-hour files).
      2. Check global data/ dirs for stale archives from previous runs.
      3. Run fetchData.sh with DATA_DIR=raw_dir.
      4. Fallback: move any archives that landed in data/ to raw_dir
         (handles unpatched fetchData.sh; only allowed with
         --allow-global-data-fallback).
      5. Save newly-downloaded NodeMetrics to cache.
      6. Log per-source archive sizes.
    """
    raw_dir   = paths.hour_raw(h)
    day, hour = from_abs(h)
    ed, eh    = from_abs(h + 1)
    start_arg = f"{day}d{hour}"
    end_arg   = f"{ed}d{eh}"

    if cfg.skip_download:
        logging.info("  --skip-download: using existing files.")
        gz_list = list(raw_dir.rglob("*.tar.gz"))
        total   = sum(f.stat().st_size for f in gz_list)
        logging.info(f"  Found {len(gz_list)} archive(s), {total/1e9:.3f} GB")
        return

    check_disk(paths.root, "download", cfg.disk_abort_gb, cfg.disk_warn_gb)

    already_cached: set[int] = set()
    if "nodemetrics" in sources:
        already_cached = populate_nm_from_cache(
            paths, raw_dir, h, cfg.no_nm_cache)

    # fetchData.sh writes to DATA_DIR, so archives already in data/ are
    # leftovers that could contaminate this hour's processing.
    stale: list[Path] = []
    for src_name in ["MSMetrics", "MSRTMCR", "NodeMetrics"]:
        stale.extend((paths.data / src_name).glob("*.tar.gz"))

    if stale:
        stale_names = [f.name for f in stale[:8]]
        msg = (
            f"Stale archives found in global data/ directories "
            f"({len(stale)} file(s): {stale_names}"
            f"{'…' if len(stale) > 8 else ''}). "
            "These could contaminate the current hour's processing. "
            "Remove them manually, or set --allow-global-data-fallback to move "
            "them into the current hour's raw dir."
        )
        if not cfg.allow_global_data_fallback:
            raise RuntimeError(msg)
        logging.warning(f"  --allow-global-data-fallback set: {msg}")

    script = paths.root / "fetchData.sh"
    if not script.exists():
        raise FileNotFoundError(f"fetchData.sh not found at {script}")

    for src_name in ["MSMetrics", "MSRTMCR", "NodeMetrics"]:
        (raw_dir / src_name).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DATA_DIR"] = str(raw_dir)

    cmd = ["bash", str(script), f"start_date={start_arg}", f"end_date={end_arg}"]
    logging.info(f"  CMD: DATA_DIR={raw_dir} {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(paths.root), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"fetchData.sh exited with code {result.returncode}")
    logging.info(f"  fetchData.sh finished in {time.time()-t0:.1f}s")

    # Archives in data/ mean fetchData.sh ignored DATA_DIR: fail unless
    # --allow-global-data-fallback is set.
    stray: list[tuple[Path, Path]] = []
    for src_name in ["MSMetrics", "MSRTMCR", "NodeMetrics"]:
        for gz in (paths.data / src_name).glob("*.tar.gz"):
            dest_sub = raw_dir / src_name
            dst = dest_sub / gz.name
            if not dst.exists():
                stray.append((gz, dst))

    if stray:
        if cfg.allow_global_data_fallback:
            for gz, dst in stray:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(gz), str(dst))
            logging.warning(
                f"  Fallback WARNING: {len(stray)} archive(s) were in data/ "
                f"instead of {raw_dir}. DATA_DIR may not have been honoured by "
                f"fetchData.sh.  Moved because --allow-global-data-fallback is set."
            )
        else:
            names = ", ".join(gz.name for gz, _ in stray[:5])
            raise RuntimeError(
                f"DATA_DIR was set to {raw_dir!r} but {len(stray)} archive(s) "
                f"landed in {paths.data!r} instead ({names}...). "
                f"This means fetchData.sh ignored DATA_DIR. "
                f"Pass --allow-global-data-fallback to allow automatic relocation, "
                f"or investigate the fetchData.sh invocation above."
            )

    if "nodemetrics" in sources:
        save_nm_to_cache(paths, raw_dir, h, cfg.no_nm_cache, already_cached)

    for src_key in sorted(sources):
        subdir, _ = SOURCE_META[src_key]
        sd = raw_dir / subdir
        if sd.exists():
            files = list(sd.glob("*.tar.gz"))
            size  = sum(f.stat().st_size for f in files)
            logging.info(f"  {subdir}: {len(files)} archive(s), {size/1e9:.3f} GB raw")


def _force_reset_hour(paths: Paths, h: int) -> None:
    """
    Remove all evidence of a previous successful commit for hour *h* so that
    run_hour() processes it from scratch when --force is given.

    Removes:
      • the _SUCCESS marker  (otherwise cleanup_partial_commit() returns early)
      • the final/ partitions for this hour (stale Parquet files)
      • the manifest entry  (so is_completed() doesn't short-circuit in the loop)

    Staging and raw dirs are NOT touched — they will be rebuilt or re-downloaded
    in the normal course of run_hour().
    """
    logging.info(f"  [--force] Resetting hour {h} before reprocessing …")

    marker = paths.success_marker(h)
    if marker.exists():
        marker.unlink()
        logging.info(f"  [--force] Removed _SUCCESS marker: {marker}")

    for table in TABLES:
        part = paths.final_part(table, h)
        if part.exists():
            shutil.rmtree(str(part))
            logging.info(f"  [--force] Removed final partition: {part}")

    # The manifest is append-only JSONL; rewrite it without the entries for h.
    if paths.completed.exists() and h in load_completed(paths):
        kept_lines: list[str] = []
        with paths.completed.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                    if rec.get("hour_id") == h:
                        continue
                except json.JSONDecodeError:
                    pass  # keep malformed lines as-is
                kept_lines.append(stripped)
        with paths.completed.open("w") as f:
            for line in kept_lines:
                f.write(line + "\n")
        logging.info(f"  [--force] Removed hour {h} from completed manifest.")

    logging.info(f"  [--force] Reset complete for hour {h}.")


def run_hour(paths: Paths, h: int, cfg: argparse.Namespace) -> bool:
    """
    Orchestrate one hour end-to-end.  Returns True on success.

    Failure guarantees:
      • staging output is NOT moved to final/
      • _SUCCESS marker is NOT written
      • completed manifest is NOT updated
      • raw archives are kept in place for debugging / retry
      • failed_hours.jsonl is appended with the error
    """
    sep = "─" * 60
    logging.info(f"\n{sep}")
    logging.info(f"  Hour {h:>4}  (day {h//24}, hour-of-day {h%24:02d})  "
                 f"id={hid(h)}")
    logging.info(sep)
    day, hr = from_abs(h)
    end_day, end_hr = from_abs(h + 1)
    logging.info(f"  start_date={day}d{hr}   end_date={end_day}d{end_hr}")
    t_start = time.time()

    paths.init(h)
    hour_handler = add_file_handler(paths.hour_logs(h) / "hour.log")

    sources = {s.strip().lower() for s in cfg.sources.split(",")}
    compr   = cfg.parquet_compression

    logging.info(f"  Sources enabled: {sorted(sources)}")
    for src_key in sorted(sources):
        names = expected_archive_names(src_key, h)
        _, prefix = SOURCE_META[src_key]
        logging.info(f"  Expected archives ({prefix}): {names}")

    svc_res: Optional[pd.DataFrame] = None
    svc_rt:  Optional[pd.DataFrame] = None

    try:
        # Must run before cleanup_partial_commit(), which returns early when
        # the _SUCCESS marker is present.
        if cfg.force:
            _force_reset_hour(paths, h)

        cleanup_partial_commit(paths, h)

        log_disk(paths.root, "before download")
        download_hour(paths, h, sources, cfg)
        log_disk(paths.root, "after download")

        if not cfg.skip_download:
            validate_downloads(paths.hour_raw(h), sources, h)
        else:
            # Even with --skip-download, confirm the files are actually there
            validate_downloads(paths.hour_raw(h), sources, h)

        stg = paths.hour_stg(h)
        if stg.exists():
            logging.info(f"  Clearing stale staging dir: {stg}")
            shutil.rmtree(str(stg))
        stg.mkdir(parents=True, exist_ok=True)

        raw = paths.hour_raw(h)
        ext = paths.hour_ext(h)

        if "msmetrics" in sources:
            logging.info("\n  [A] MSMetrics")
            t0 = time.time()
            svc_res = process_msmetrics(raw, ext, h, cfg)
            out = paths.staging_part("service_resource", h) / "part.parquet"
            save_parquet(svc_res, out, compr)
            logging.info(f"  service_resource staging: {out.stat().st_size/1e6:.1f} MB  "
                         f"elapsed {time.time()-t0:.1f}s")
            log_disk(paths.root, "after MSMetrics")

        if "msrtmcr" in sources:
            logging.info("\n  [B] MSRTMCR")
            t0 = time.time()
            svc_rt = process_msrtmcr(raw, ext, h, cfg)
            out = paths.staging_part("service_rtmcr", h) / "part.parquet"
            save_parquet(svc_rt, out, compr)
            logging.info(f"  service_rtmcr staging: {out.stat().st_size/1e6:.1f} MB  "
                         f"elapsed {time.time()-t0:.1f}s")
            log_disk(paths.root, "after MSRTMCR")

        if "nodemetrics" in sources:
            logging.info("\n  [C] NodeMetrics")
            t0 = time.time()
            node = process_nodemetrics(raw, ext, h, cfg)
            out  = paths.staging_part("node_metrics", h) / "part.parquet"
            save_parquet(node, out, compr)
            logging.info(f"  node_metrics staging: {out.stat().st_size/1e6:.1f} MB  "
                         f"elapsed {time.time()-t0:.1f}s")
            del node
            log_disk(paths.root, "after NodeMetrics")

        if svc_res is not None:
            logging.info("\n  [D] Joined service features")
            t0 = time.time()
            joined = build_joined(svc_res, svc_rt)
            out    = paths.staging_part("joined_service_features", h) / "part.parquet"
            save_parquet(joined, out, compr)
            logging.info(f"  joined_service_features staging: {out.stat().st_size/1e6:.1f} MB  "
                         f"elapsed {time.time()-t0:.1f}s")
            del joined

        logging.info("\n  [E] Validation")
        stats = validate_staging(paths, h)

        logging.info("\n  [F] Committing to final/")
        commit_hour(paths, h)
        log_disk(paths.root, "after commit")

        elapsed = round(time.time() - t_start, 1)
        write_completed(paths, h, {
            "day":          h // 24,
            "hour_of_day":  h %  24,
            "start_date":   f"{day}d{hr}",
            "end_date":     f"{end_day}d{end_hr}",
            "sources":      sorted(sources),
            "elapsed_s":    elapsed,
            **stats,
        })
        logging.info(f"\n  ✓ Hour {h} completed in {elapsed}s")

        if cfg.cleanup_on_success:
            cleanup_hour(paths, h,
                         keep_raw=cfg.keep_raw,
                         keep_extracted=cfg.keep_extracted)
            log_disk(paths.root, "after cleanup")

        return True

    except Exception as exc:
        logging.error(f"\n  ✗ Hour {h} FAILED: {exc}", exc_info=True)
        write_failed(paths, h, str(exc))
        logging.info("  Raw archives retained for debugging/retry.")
        return False

    finally:
        remove_handler(hour_handler)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_hourly_pipeline.py",
        description="Alibaba Trace v2022 — hourly preprocessing pipeline (hardened)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--start", required=True,
                   help="Start hour inclusive, e.g. 0d0")
    p.add_argument("--end",   required=True,
                   help="End hour exclusive, e.g. 0d1")
    p.add_argument("--project-root", default=".",
                   help="Project root containing fetchData.sh")

    p.add_argument("--cleanup-on-success",
                   type=lambda x: x.lower() in ("true", "1", "yes"),
                   default=True,
                   help="Delete raw+extracted files after success (default: true)")
    p.add_argument("--force", action="store_true",
                   help="Reprocess hours already in the completed manifest")
    p.add_argument("--keep-raw", action="store_true",
                   help="Keep raw archives even with --cleanup-on-success")
    p.add_argument("--keep-extracted", action="store_true",
                   help="Keep extracted CSVs even with --cleanup-on-success")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download; use files already in work/hour_*/raw/")
    p.add_argument("--sources", default="MSMetrics,MSRTMCR,NodeMetrics",
                   help="Comma-separated sources (default: all three)")
    p.add_argument("--parquet-compression", dest="parquet_compression",
                   default="zstd", choices=["zstd", "snappy", "gzip", "none"],
                   help="Parquet compression (default: zstd)")

    p.add_argument("--disk-abort-gb", type=float, default=25.0,
                   help="Hard-abort if free disk < N GB (default: 25)")
    p.add_argument("--disk-warn-gb",  type=float, default=40.0,
                   help="Warn if free disk < N GB (default: 40)")
    p.add_argument("--mem-abort-pct", type=float, default=90.0,
                   help="Hard-abort if RAM usage > N%% (default: 90)")

    p.add_argument("--allow-global-data-fallback", action="store_true",
                   help="Allow moving archives from data/ to raw/ if DATA_DIR "
                        "was not honoured by fetchData.sh. USE WITH CAUTION: "
                        "stale files in data/ will be silently included.")
    p.add_argument("--no-nm-cache", action="store_true",
                   help="Disable the NodeMetrics archive cache "
                        "(cache/NodeMetrics/). Re-downloads every 12-hour block.")

    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging")
    return p


def main() -> None:
    args  = build_parser().parse_args()
    root  = Path(args.project_root).resolve()
    paths = Paths(root)

    setup_logging(paths.pipe_logs / "pipeline.log", args.verbose)

    logging.info(f"{'='*60}")
    logging.info(f"  Alibaba Trace Pipeline  v{VERSION}")
    logging.info(f"{'='*60}")
    logging.info(f"  project root      : {root}")
    logging.info(f"  dataframe library : {'polars' if _POLARS else 'pandas'}")
    logging.info(f"  memory guard      : "
                 f"{'active (psutil)' if _PSUTIL else 'inactive — install psutil'}")
    logging.info(f"  parquet codec     : {args.parquet_compression}")
    logging.info(f"  disk abort/warn   : {args.disk_abort_gb} / {args.disk_warn_gb} GB free")
    logging.info(f"  mem abort         : > {args.mem_abort_pct}% RAM used")
    logging.info(f"  NodeMetrics cache : "
                 f"{'disabled (--no-nm-cache)' if args.no_nm_cache else paths.cache_dir}")
    logging.info(f"  global data fallback: "
                 f"{'ALLOWED (--allow-global-data-fallback)' if args.allow_global_data_fallback else 'denied (default)'}")

    sd, sh  = parse_date(args.start)
    ed, eh  = parse_date(args.end)
    h_start = to_abs(sd, sh)
    h_end   = to_abs(ed, eh)

    if h_end <= h_start:
        logging.error(f"--end ({args.end}) must be strictly after --start ({args.start})")
        sys.exit(1)

    hours = list(range(h_start, h_end))
    logging.info(f"\n  Range   : {args.start} → {args.end}  ({len(hours)} hour(s))")
    logging.info(f"  Sources : {args.sources}\n")

    # Validate source names before any work begins.
    requested_sources = {s.strip().lower() for s in args.sources.split(",")}
    unknown = requested_sources - set(SOURCE_META.keys())
    if unknown:
        hint = ""
        if "callgraph" in unknown or "call_graph" in unknown:
            hint = (
                " Note: CallGraph is intentionally excluded from this pipeline "
                "because it is extremely large and not needed for autoscaling analysis."
            )
        logging.error(
            f"Unknown source(s): {sorted(unknown)}. "
            f"Valid sources are: {sorted(SOURCE_META.keys())}.{hint}"
        )
        sys.exit(1)

    paths.manifests.mkdir(parents=True, exist_ok=True)

    succeeded  = 0
    skipped    = 0
    failed_hrs: list[int] = []

    for h in hours:
        if not args.force and is_completed(paths, h):
            logging.info(f"Hour {h} already completed — skipping "
                         "(use --force to reprocess).")
            skipped += 1
            continue

        ok = run_hour(paths, h, args)
        if ok:
            succeeded += 1
        else:
            failed_hrs.append(h)

    sep = "=" * 60
    logging.info(f"\n{sep}")
    logging.info("  Pipeline summary")
    logging.info(sep)
    logging.info(f"  Total   : {len(hours)}")
    logging.info(f"  Done    : {succeeded}")
    logging.info(f"  Skipped : {skipped}")
    logging.info(f"  Failed  : {len(failed_hrs)}")
    if failed_hrs:
        logging.info(f"  Failed hours: {failed_hrs}")
    log_disk(paths.root, "final")

    state = {
        "pipeline_version": VERSION,
        "last_run":  datetime.now(timezone.utc).isoformat(),
        "start":     args.start,
        "end":       args.end,
        "succeeded": succeeded,
        "skipped":   skipped,
        "failed":    failed_hrs,
    }
    with paths.state.open("w") as f:
        json.dump(state, f, indent=2)
    logging.info(f"  State written → {paths.state}")

    sys.exit(0 if not failed_hrs else 1)


if __name__ == "__main__":
    main()
