# Release notes for analysis/adaptive

The seven code files of the v1 freeze (`manifest_v1.json`) are byte-identical to the versions that ran the v1 replay, so `verify_manifest.py` and `run.py` accept them. The frozen `adaptive/data.py` has a session path as its default repository root, so `OBA_REPO` must be set. The frozen `tests.py` reads the exploratory outputs from a session path in its last check.

Files added after the v1 run, for reporting and packaging only (controllers not re-run):

- `tests_portable.py`: `tests.py` with the last check made optional (`OBA_V0_DIR`).
- `reproduce.sh`, `compare_outputs.py`: re-run in a separate folder and comparison with `out_v1/`.
- `summarise.py` (rewritten): `infinite_set_share` is the share of unit-minutes with an infinite ACI set (the v1 field of that name was the mean number of such minutes per unit, now `mean_infinite_minutes`); units whose budget is attainable at constant capacity `M_u` (post hoc diagnostic); pooled summaries of the Huawei periods; mean resource, churn and penalty parts.
- `make_tables.py`: LaTeX tables of Additional file 1, Sect. S12.
- `verify_manifest.py`, `reader_qa.py`, `README.md`, this file, `manifest_release.json`.
