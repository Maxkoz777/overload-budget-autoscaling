# Adaptive comparators: BACC-r, ACI and ACI-G (Sect. 6.4 of the article; Additional file 1, Sect. S12)

This folder compares PAC-h with three adaptive controllers under the closed-loop replay of the article:

- **BACC-r**, our re-implementation of the controller described in arXiv:2606.20575v2 (BACC). It is not the published system: forecasts, the replica profile and the actuation queue are those of this replay. `SOURCE_MAPPING.md` maps every operation to the section or equation of v2.
- **ACI**, adaptive conformal inference (Gibbs and Candès, 2021) with `gamma = 0.005` and target `delta`, run as `tau + 1` interleaved lanes so that each update uses an observed error.
- **ACI-G**, the same algorithm with a finite-horizon target `alpha* = delta - (tau+1)(1+gamma)/(gamma T)`, so that its formal miscoverage over the test period is at most `delta` for every unit. `ACI_BOUND.md` derives the bound for the implemented algorithm and lists when actual overload can exceed formal errors.

Every controller, including PAC-h, is replayed under a common resource bound `M_u = max(1, ceil(2 max pre-period demand / mu))`. PAC-h is also reported without the bound, as the bridge to the other results of the article. Under the bound PAC-h is a resource-limited variant: clipping can break `mu k >= forecast + margin`, so the coverage statements of the article do not transfer to it automatically.

## Layout

| Path | Content |
|---|---|
| `adaptive/` | data access (reuses `analysis/audit/pac_window_grid.py` and `confirmatory_replay.py`), actuation queue, controllers |
| `protocol_v1.json` | protocol of the v1 run, including the gamma rule and its revision R1 |
| `manifest_v1.json`, `manifest_v1.sha256` | SHA-256 freeze written before the v1 test run (self-recorded) |
| `tests.py`, `tests.log` | checks run before the v1 test run (frozen) |
| `tests_portable.py`, `tests_release_2026-11-09.log` | the same checks with portable paths (68 passed in the packaging check with `OBA_V0_DIR` set; 67 passed and the v0 check skipped on macOS) |
| `reproduce.sh`, `compare_outputs.py` | re-run in a separate folder and comparison with the frozen outputs |
| `test_compare_outputs.py` | checks that `compare_outputs.py` rejects missing, duplicated, changed or non-finite results |
| `manifest_release.json` | SHA-256 of every file of this folder as released |
| `run.py` | `cal` (gamma selection on days 8-9), `test` (three traces), `confirm` (Huawei periods, post hoc) |
| `summarise.py` | cell summaries from the per-unit results (reporting only) |
| `verify_manifest.py` | preflight against the frozen manifest (the same check `run.py` makes) |
| `reader_qa.py` | the fallback parquet reader against pyarrow |
| `make_tables.py` | LaTeX tables of Additional file 1, Sect. S12 (`out_v1/tables_S12.tex`) |
| `out_v1/` | per-unit results, calibration files, summaries |
| `exploratory_v0/` | the earlier exploratory runs, kept unchanged; superseded by v1 |

## Reproduce

The prepared series must be in `experiments/data/` of the repository, as described in `docs/DATA.md` (`service_timeseries_200_verified/` for Alibaba, `external_traces/` for Huawei and Azure). The frozen code locates them through `OBA_REPO`, the repository root. From this folder:

```bash
export OBA_REPO="$(cd ../.. && pwd)"
python3 verify_manifest.py      # frozen code, protocol, parameters and the 222 data inputs against manifest_v1.json
python3 tests_portable.py       # the checks of tests.py with portable paths
bash reproduce.sh               # cal, test and confirm in a new folder rerun_<time>/, then compare_outputs.py
```

`reproduce.sh` re-runs the step-size selection, keeps it for comparison, and runs the test and the Huawei periods with the frozen calibration files, because `run.py` refuses to start when a frozen code or calibration file differs from the manifest. `compare_outputs.py` then compares the re-run with the frozen `out_v1/`: it requires all four result files, the same columns and unique keys, the same rows, equal missing-value masks and text values in every column, numeric values within a relative tolerance of 1e-9, finite outcome metrics, and equal counts of units within budget in every cell. `test_compare_outputs.py` checks that it rejects incomplete or altered results. `summarise.py` and `make_tables.py` rebuild the summaries and the tables of Additional file 1, Sect. S12, from `out_v1/`.

The parquet files are read with pyarrow when it is installed, otherwise with `adaptive/pq.py`, which `reader_qa.py` checks against pyarrow (200 files x 4 columns, no difference). `tests.py` is the frozen version that ran before the v1 test replay; its last check reads the exploratory outputs from the session path of that run, so use `tests_portable.py` elsewhere (set `OBA_V0_DIR=exploratory_v0` to include that check).

## Status of the analyses

- v0 (`exploratory_v0/`): exploratory, superseded. An independent review found that its early trigger bypassed the change limit and that its ACI diagnostics were shifted under delay.
- v1 main replay: protocol and code frozen before the test run. The primary gamma rule leaves some cells without an admissible gamma; revision R1, written after the v0 results were known, fills them and is labelled in every output.
- Huawei periods: the same rules on the three periods of the confirmatory replay, a post hoc baseline analysis, not part of the frozen confirmatory test.
