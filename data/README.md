# Data

`upstream/` contains the three exact local CSVs used by the historical study.
Do not normalize or edit them. Their hashes and licensing caveat are recorded in
`artifacts/manifests/artifacts.json` and `DATA_LICENSE.md`.

`canonical/combined_full.csv` is deterministically generated from those files.
It contains 15,526 daily rows from 1981-06-29 through 2023-12-31, seven raw
features, and causal trailing means at 30, 60, 90, 120, 150, and 180 days.

Rebuild to a separate path:

```powershell
python scripts/rebuild_dataset.py --output tmp/combined_full.csv
```

The command fails if the generated SHA-256 differs from the canonical hash.

