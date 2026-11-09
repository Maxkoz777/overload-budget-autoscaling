#!/usr/bin/env bash
# Re-run the adaptive comparison in a separate folder and compare it with the frozen outputs.
# Usage (from this folder):  OBA_REPO=<repository root> bash reproduce.sh [<work folder>]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${OBA_REPO:?set OBA_REPO to the repository root (the folder that contains analysis/ and experiments/)}"
export OBA_REPO
WORK="${1:-$HERE/rerun_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$WORK/out_v1"
cp -R "$HERE/adaptive" "$HERE/run.py" "$HERE/tests.py" "$HERE/manifest_v1.json" "$HERE/protocol_v1.json" \
      "$HERE/SOURCE_MAPPING.md" "$HERE/ACI_BOUND.md" "$WORK/"
cd "$WORK"
python3 --version | tee environment.txt
python3 -c "import numpy, pandas, scipy; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__)" | tee -a environment.txt
python3 -c "import importlib.util as u; print('pyarrow', __import__('pyarrow').__version__ if u.find_spec('pyarrow') else 'not installed')" | tee -a environment.txt
python3 -c "from adaptive.data import READER; print('parquet reader', READER)" 2>/dev/null | tee -a environment.txt
python3 "$HERE/verify_manifest.py" | tee preflight.log
python3 run.py cal | tee cal.log
mv out_v1/cal_traces.csv out_v1/cal_traces_rerun.csv
mv out_v1/cal_periods.csv out_v1/cal_periods_rerun.csv
cp "$HERE/out_v1/cal_traces.csv" "$HERE/out_v1/cal_periods.csv" out_v1/
python3 run.py test | tee test.log
python3 run.py confirm | tee confirm.log
python3 "$HERE/compare_outputs.py" out_v1 "$HERE/out_v1" | tee comparison.log
