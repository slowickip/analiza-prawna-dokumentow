# Rights for the evaluation material

## CC0 dedication for the project's own material

To the extent that copyright, database rights or related rights exist in work the project author
created under this directory, the project author dedicates that work to the public domain under the
Creative Commons CC0 1.0 Universal dedication.

SPDX identifier: `CC0-1.0`

Canonical legal text: https://creativecommons.org/publicdomain/zero/1.0/legalcode

This dedication covers the clean-room synthetic documents under `evaluation-data/documents/` and the
project's own records (`manifest.json`, `README.md`, `LICENSE.md`,
`results/development/ca-009-live-validation.json` and `source-files/acquisition.json`). The two
document files under `evaluation-data/source-files/` are third-party material and are not covered by
this dedication; the section below states the terms they are carried under, and `manifest.json`
records their rights status.

The synthetic documents are research fixtures, not legal templates or legal advice.

## Third-party source documents

`evaluation-data/source-files/` holds two contract templates published by the Polish Ministry of
Development and Technology:

| File | Published as |
| --- | --- |
| `RT-001-housing-cooperative.doc` | *Wzór umowy kooperatywy mieszkaniowej* |
| `RT-002-civil-partnership.doc` | *Wzór umowy spółki cywilnej* |

The templates are redistributed under the ministry's
[conditions for free re-use of public sector information](https://www.gov.pl/web/rozwoj-technologia/uzyskaj-informacje-publiczna-do-ponownego-wykorzystania).
These require source and timing information, a description of processing and, for protected works,
attribution to the creator where known. Keep the following attribution with copies of the templates:

- **Source:** Ministerstwo Rozwoju i Technologii,
  https://www.gov.pl/web/rozwoj-technologia/kooperatywy-mieszkaniowe
- **Credits:** the source page attributes the drafting to ministry experts. Both published files
  also carry the following metadata, preserved here under the publisher's labels:
  - `Author`: **Jacek Mościcki**.
  - `Last saved by`: **Szpot-Prusak Katarzyna**; this name also appears in the editing history.
  - `Company`: **Wydawnictwo C.H.Beck sp. z o.o.**
- **File creation and last-save times:** the table below transcribes the original file metadata.
- **Time obtained:** 2026-08-30, by download from the page above;
  `source-files/acquisition.json` records the attachment URL, the byte count and the hash of what the
  publisher served.
- **Processing applied:** the files are renamed and author/editor strings in the document metadata
  are overwritten with blank filler of the same length. Attribution is retained in this notice,
  outside the documents passed to the system. The document-content streams are byte-identical to
  the originals. `source-files/acquisition.json` records the metadata, the verification and separate
  hashes for the original download and the stored derivative.
- The publisher states that the templates are not binding and that a reader may use one and adapt its
  provisions. The publisher takes no responsibility for public sector information a re-user has
  processed.

| File | Created (UTC) | Last saved (UTC) |
| --- | --- | --- |
| `RT-001-housing-cooperative.doc` | 2023-10-24 08:01 | 2023-11-14 10:35 |
| `RT-002-civil-partnership.doc` | 2023-10-24 08:08 | 2023-10-24 08:14 |

The original downloads were checked against the recorded acquisition hashes. The attribution above
preserves the original metadata labels for each credit.
Together with the source, timestamps and processing notice, it fulfils the ministry's stated re-use
conditions for these copies. `manifest.json` records the verified basis for their redistribution.

The files are research fixtures here, exactly as the synthetic documents are. They are not legal
advice, and a reader who wants a template to use should take it from the publisher's page.

## Statutory source texts

The runtime search corpus is rebuilt from `corpus/manifest.example.json` through the publisher's API.
Offline evaluation also ships statutory text in `judge/sources/provisions.json`: 3317 provisions
from eleven acts, with ELI references and content hashes. These official source texts are separate
from the project's own material covered by the CC0 dedication above.
