# SimulacraBench entry (NeurIPS 2026)

- `submissions/v3_ens/main.py` — main submission (multi-task nets + count backoff, CV stacking, gate rules). Single file, no extra requirements.
- `submissions/v1_backoff/` — organizer count-backoff baseline without the LLM prior (control).
- `zips/` — upload-ready archives (built with `cd submissions/X && zip -r ../../zips/X.zip .`).
- `tools/eval_local.py` — exact (un-noised) local scoring; `tools/make_sandbox_v2.py` — a second synthetic generator.
- `results/` — local score logs (JSON lines). `writeup/report.tex` — draft technical report.

Setup: `git clone https://github.com/SituatedEvals/public starter-kit` (tested at aa03186), then
`python starter-kit/make_sandbox.py --schema starter-kit/data/unicef.json --out starter-kit/_sandbox/unicef` (same for unhcr, world_bank) and
`python tools/eval_local.py --submission submissions/v3_ens`.

License: MIT.
