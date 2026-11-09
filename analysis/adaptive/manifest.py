import hashlib, json, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy
from adaptive.data import REPO, READER

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
code = ["adaptive/__init__.py", "adaptive/data.py", "adaptive/env.py", "adaptive/policies.py", "adaptive/pq.py", "run.py", "tests.py"]
data_root = REPO / "experiments" / "data"
inputs = {}
for p in sorted(list((data_root / "service_timeseries_200_verified").glob("*.parquet")) + list((data_root / "external_traces").rglob("*.*")) + list((data_root / "splits").glob("*"))):
    inputs[str(p.relative_to(data_root))] = sha(p)
audit = {p.name: sha(p) for p in sorted((REPO / "analysis" / "audit").glob("*.py"))}
m = dict(note="self-recorded freeze, not an external preregistration",
         protocol={"protocol_v1.json": sha("protocol_v1.json"), "SOURCE_MAPPING.md": sha("SOURCE_MAPPING.md"), "ACI_BOUND.md": sha("ACI_BOUND.md")},
         code={c: sha(c) for c in code}, repository_audit_modules=audit, tests_log={"tests.log": sha("tests.log")},
         selected_parameters={f: sha(f"out_v1/{f}") for f in ("cal_traces.csv", "cal_periods.csv")},
         data_inputs=inputs, environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, parquet_reader=READER))
Path("manifest_v1.json").write_text(json.dumps(m, indent=1) + "\n")
print("manifest entries:", len(inputs), "inputs;", sha("manifest_v1.json"))
