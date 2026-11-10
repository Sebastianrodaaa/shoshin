# NOTICE

shoshin's own code is MIT (see `LICENSE`). It builds on the work below, which keeps its own terms.
shoshin is not affiliated with, endorsed by, or sponsored by the ARC Prize Foundation, and uses none of its marks
or badges.

## WARC (code and weights) — CC0-1.0

- Code: https://github.com/third-intelligence/WARC (CC0-1.0). shoshin imports it unchanged from
  `third_party/WARC` (`warc_model.model.build_from_checkpoint`, `warc_model.final_ttt.FinalPredictor`, its Muon
  optimizer, its recording parser and window items). The one behavioural difference is FP32 instead of bf16 on CPU
  (`shoshin/model.py`).
- Weights: https://huggingface.co/third-intelligence/WARC (CC0-1.0). Only the three held-out fold checkpoints are
  used for results; `all25` is never used.
- Paper: Onoda, Kawano, Takashiro, Hayashi. *WARC.* arXiv:2611.12324, https://arxiv.org/abs/2611.12324 and
  https://huggingface.co/papers/2611.12324.
- VARC and the other third-party components vendored inside WARC keep their own licences; see WARC's
  `docs/third_party.md`.

## ARC-AGI-3 public games and human recordings — ARC Prize Foundation

The public ARC-AGI-3 games and the Public Demo human recordings belong to the ARC Prize Foundation
(https://arcprize.org). Terms check, done 2026-11-10:

- https://arcprize.org/terms (updated 2024-06-03) restricts the website's "Services" Content to personal /
  non-commercial or internal business use and requires written permission to publicly display that site Content
  for commercial purposes. shoshin does not scrape or republish site chrome.
- That is not a ban on the separately open-sourced public tasks. https://arcprize.org/policy: "Public Tasks - Fully
  open source and available for anyone to use. These are published in our GitHub repositories and are intended for
  research, development, and community experimentation."
- The official toolkit (`arc-agi` / `arcengine`) is MIT. The dataset post
  (https://arcprize.org/blog/arc-agi-3-human-dataset) says the foundation is "open sourcing the full Public Demo
  dataset" (step-by-step replays of the 25 public environments).
- No separate SPDX licence file for the recordings themselves was found on the official pages.

shoshin uses public games and public recordings only, never private or semi-private tasks. Frames of public games
appear in this research demo, the README GIF and the demo footage under the public-task terms above. The repository
carries a small subset of real recordings (listed in `data/README.md`, each checked against WARC's
`data/SHA256SUMS`), not the full corpus. If a source later forbids redistribution, shoshin stops redistributing
frames and says so here.

## Banner engraving — public domain

Albrecht Dürer, *Melencolia I*, 1514, engraving. Public domain. Source scan: National Gallery of Art,
Washington, 1943.3.3522 (open access), via Wikimedia Commons
(https://commons.wikimedia.org/wiki/File:Albrecht_D%C3%BCrer_-_Melencolia_I_(NGA_1943.3.3522).jpg). Dithered to
two tones (Floyd-Steinberg) for `docs/screenshots/banner.png` by `scripts/banner.py`.

## Fonts

- Silkscreen by Jason Kottke, SIL Open Font License 1.1 (dashboard numbers and name; `app/silkscreen.ttf`).
- UnifrakturMaguntia by j. 'mach' wust, SIL Open Font License 1.1 (banner title only, rendered into the PNG).
- JetBrains Mono, SIL Open Font License 1.1 (banner tagline only, rendered into the PNG).

## UI references

The Pinterest pins cited in `docs/brief/ui-refs/REFS.md` belong to their authors and are kept for reference only.
