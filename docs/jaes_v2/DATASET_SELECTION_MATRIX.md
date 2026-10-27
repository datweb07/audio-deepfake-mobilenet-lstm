# Public dataset selection and acquisition status

Sources inspected 25 October 2026. Unknown values are intentional. Full archives were not downloaded.

| Dataset | Official release / version | License | Compressed / expanded size | REAL / FAKE coverage | Identity and partition metadata | Intended role / feasibility |
|---|---|---|---|---|---|---|
| ASVspoof 2019 LA | [Edinburgh release](https://datashare.ed.ac.uk/handle/10283/3336), DOI 10.7488/ds/2555 | ODC-By 1.0 per release README; preserve license/content conditions | LA.zip listed 7.12 GB; expanded size NOT_VERIFIED | Official train 2,580 / 22,800; other counts computed from actual protocol before locking | Official speaker and attack-condition fields; train/dev/eval. Exact generator/source-parent mapping not guaranteed | **Primary importer target**. Full compressed+expanded+cache budget not approved by default; prefer an existing local extracted copy or separately approved bounded subset |
| In-the-Wild | [Author release](https://deepfake-total.com/in_the_wild), linked current hosting | Apache 2.0 per author page | Hosting reports 8.16 GB total; expanded size NOT_VERIFIED | Both classes; exact recording counts NOT_VERIFIED here | 58 named figures; exact generator identities not established; intended evaluation collection | Future independent-corpus transfer diagnostic; not verified unseen-generator evaluation |
| WaveFake | [Zenodo 5642694](https://zenodo.org/records/5642694), v1.2.0 | CC-BY-SA 4.0 per release | 28.9 GB generated-audio archive; expanded size NOT_VERIFIED | 104,885 generated clips; reference REAL audio not redistributed there | Generator families and LJSpeech/JSUT lineage described; small speaker diversity; custom split needs audit | Useful future generator/lineage study; generated archive plus originals/extraction exceeds a prudent default for ~40 GB free disk |
| ASVspoof 5 | [Zenodo 14498691](https://zenodo.org/records/14498691), v0.1 listing | Read release LICENSE before acquisition; exact terms NOT_VERIFIED in this phase | 142.3 GB listed files; protocols archive 20.7 MB; expanded size NOT_VERIFIED | Both classes; counts not transcribed without parsing protocols | Rich official attack/speaker/task protocols | Metadata review only; **no default full download** |

The ASVspoof evaluation plan and release README disagree on some preliminary speaker-count descriptions. The importer therefore computes actual identities from the official protocol rather than hard-coding a speaker count. Attack IDs identify challenge conditions; they are not automatically exact generative-model identities. Official 2019 train/eval are splits of one corpus, not independent corpora.

## Actual acquisition

Small-license retrieval was attempted from the Edinburgh official endpoint. The sandbox first failed DNS; the approved unsandboxed retry reached the host but received HTTP 403. No authentication, access-control bypass, bulk archive or unofficial mirror was used. **Audio downloaded: 0 bytes. Protocol/license files downloaded: 0 bytes.** The official web source was inspectable through the research browser; generated audit notes are not downloaded raw protocol files.

Status: **AWAITING_DATA_ACQUISITION**. Actual audio integration remains pending. The importer and protocol fixtures are implementation tests, never an external benchmark. This activates the user's allowed legacy-development sensitivity branch for the current bounded run.
