# Publication Repository Instructions

This repository reproduces the submitted OCEANS 2026 Lake Superior
water-level forecasting paper. Scientific equivalence to the frozen experiment
takes priority over architectural cleanup.

Before changing scientific code, inspect `README.md`, `docs/methods.md`,
`docs/provenance.md`, the active configuration, and the relevant reproduction
tests. Never select models or tune behavior using test metrics.

The historical GLM repository is external, read-only provenance. This package
must not import from GLM or depend on GLM paths.

Non-negotiable contracts:

- canonical `ofs1` condition identifiers remain compatible with the historical
  implementation;
- preprocessing is causal and split assignment uses target timestamps;
- final model selection is validation-only;
- the manuscript evaluation surface is exactly 2,926 forecast origins from
  2015-07-01 through 2023-07-04;
- RQ3 manuscript analysis is derived from raw fixed-refit predictions, not the
  historical 2,956-origin aggregate products;
- paper reproduction must not import training or distributed dependencies;
- CPU, CUDA, and MPS are supported without hostname restrictions;
- upstream data are not covered by the repository MIT code license.

Generated artifacts must have manifests and must not overwrite frozen inputs.
Run the focused tests and `python scripts/reproduce_paper.py` before changing a
scientific golden result.

