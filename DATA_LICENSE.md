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
`b55b64d4b08dd5ca0295846b93eb4483`, so it was not archive-byte-identical to
the current Zenodo ZIP. On 2026-08-28, the current official archive was
downloaded from the Zenodo record and verified as:

- filename: `Dual-Transformer-1.0.zip`;
- byte size: `84,421,571`;
- MD5: `cf97c772ff861d0779b5cff1ccee0a2f`;
- SHA-256: `a2d0fb81c93deda910594af61bb1f2477a10b87d4b4d820d9ed2ae279f5bbffd`.

The three Lake Superior members in that official archive are byte-identical to
the preserved experiment inputs. Their SHA-256 values are:

- `finaldata.csv`: `98e21a38cecbbbe629ced269551823bf448caa2f0413b57cb804beed9cdbf053`;
- `test_2022.csv`: `c0b4b45ed75ffea073e0aad118139a0e88084aa205ea7fa627d96377d54b5511`;
- `test_2023.csv`: `c7e66a43c7f0f539c186b3a05921420fe255ff80a347c7bfd743ee172f29de2c`.

This establishes current external member-level provenance for the exact three
inputs used by the experiment. It does not imply that the older local ZIP was
otherwise identical to the current deposit.

Frozen experiment exports are provided for scientific verification. Their
presence does not change the licensing of upstream data embedded in their
values. The paper source remains the authors' submitted manuscript and is not
covered by the software MIT license unless separately stated.
