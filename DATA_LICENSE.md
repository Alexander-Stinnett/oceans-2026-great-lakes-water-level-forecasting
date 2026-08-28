# Data and frozen-artifact licensing

The MIT license in `LICENSE` applies to original code in this repository. It
does not relicense upstream data, pretrained models, frozen experiment exports,
or the submitted manuscript.

The three files in `data/upstream` were obtained for the historical study from
the Dual Transformer release identified as Zenodo record
`10.5281/zenodo.15276228`. The record is labeled GNU GPL v3.0-or-later. The
archive also contains a GPLv3 license file. These CSVs retain their exact local
bytes and are not represented as MIT-licensed.

The Zenodo record combines software, models, and data but does not separately
state the terms governing underlying NOAA, GLSEA, or other source observations.
Users are responsible for confirming upstream terms for redistribution or reuse.
No claim beyond the record-level license is made here.

During migration, the local source archive had MD5
`b55b64d4b08dd5ca0295846b93eb4483`, while the then-current Zenodo record
reported `cf97c772ff861d0779b5cff1ccee0a2f` for its ZIP. The local archive was
therefore not archive-byte-identical to the current deposit. The embedded local
CSV members exactly match the files preserved here; comparison against freshly
downloaded current Zenodo members remains an external provenance check.

Frozen experiment exports are provided for scientific verification. Their
presence does not change the licensing of upstream data embedded in their
values. The paper source remains the authors' submitted manuscript and is not
covered by the software MIT license unless separately stated.

