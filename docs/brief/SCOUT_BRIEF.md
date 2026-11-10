# shoshin: scout brief (self-contained copy)

Source: the 2026-11-10 midday ai-frontier scout entry for shoshin and the build request that followed (Sebastian's go,
2026-11-10, Paris). The original box paths (`/workspace/ai-frontier/projects.md`,
`/workspace/ai-frontier/reports/2026-11-10-midday.md`) were not available on the build machine, so this file is
reconstructed from the build request, which quotes the brief. It is the reference for this build.

## Idea

A 35M-parameter WARC world model learns an ARC-AGI-3 game it has never seen, live, on CPU. WARC's default recipe
takes 20 gradient steps after every transition, including moves it already predicts perfectly; that is a GPU
recipe. shoshin makes learning proportional to surprise. Before each move the model commits its predicted next
frame. After the move it compares prediction and reality on the cells that changed. If it was right, no update. If it
was wrong, it runs a short update on that transition plus a small replay buffer of earlier surprises (so it does not
forget rules it already learned), with the step count scaled by how many changed cells it got wrong. The surprise
trace is also the demo: you watch it get surprised, then watch it learn.

## Demo (build first; the eval backs it)

A local dashboard of a real run, not mock screens:
- Three equal square panels: the real unseen game frame, the model's predicted next frame, the surprise map.
- Wrong cells glow hot magenta and cool to pale lilac once that rule is learned.
- A full-width rule ledger strip that ticks as rules are learned ("rule 4 learned after 3 tries").
- A narrow right rail with a compute meter: gradient steps spent vs the full every-move WARC recipe, real numbers.
- "Play yourself": the viewer takes the arrow keys (and clicks if the game uses them) and tries to surprise it.
- Something visibly changes in the first 3 seconds; readable with the sound off.
- Light theme (warm paper cream, ink black, one hot magenta cooling to lilac); pixel bitmap face for numbers and the
  name, plain grotesque for labels. See `ui-refs/`.

## Method and experiment

CPU only (torch CPU build; no GPU, CUDA index, ZeroGPU, paid compute or paid APIs). Reuse WARC
(https://github.com/third-intelligence/WARC, CC0-1.0; weights https://huggingface.co/third-intelligence/WARC, CC0-1.0;
paper arXiv:2611.12324).

Fold checkpoints and their held-out games (official model card; `models/manifest.json`). Each game is evaluated only
with the fold that held it out. `all25` saw every public game and is not used for the held-out test.

| checkpoint | held-out games | val exact-frame |
| --- | --- | --- |
| fold1 | bp35 lf52 lp85 r11l re86 sb26 sc25 su15 wa30 | 85.7% |
| fold2 | cd82 cn04 dc22 m0r0 sk48 tn36 tr87 tu93 | 79.2% |
| fold3 | ar25 ft09 g50t ka59 ls20 s5i5 sp80 vc33 | 78.5% |

Data: the ARC-AGI-3 Public Demo human recordings (ARC Prize Foundation,
https://arcprize.org/blog/arc-agi-3-human-dataset), verified against WARC's `data/SHA256SUMS`. Offline replay must
work with no API key. Live play against the ARC Prize API is optional (`ARC_API_KEY` from the environment, never
committed). Public games and public recordings only.

Arms: (1) frozen; (2) full WARC recipe (20 steps every transition); (3) random-skip at matched update count;
(4) surprise-gated; (5) surprise-gated without the surprise replay buffer.

Hypothesis (falsifiable): replaying the public human recordings of each held-out game through the fold checkpoint
that did not train on it, surprise-gated learning (1) keeps changed-cell accuracy within 2 points absolute of the full
recipe, (2) while spending at most 25% of its gradient steps, (3) beats random-skip with the same number of updates,
and (4) runs at 1 or more transitions per second on the CPU box. It fails if (1) or (3) does not hold over 3 seeds
(95% bootstrap CIs). Negative results ship. No invented numbers.

Metrics: changed-cell accuracy per transition; exposures until a recurring change pattern is predicted at 80% or more
(unsupervised proxy: action plus local change signature, labelled as a proxy until WARC releases MechEval
annotations); gradient steps; wall-clock per transition; 3 seeds. Start with a ~20-minute smoke config on one
held-out game at each arm, then size the full plan from the measured time.

## Deliverables

`shoshin/{model,gate,replay,run}.py`, `scripts/{plots,record_demo}.py`, `app/{server.py,index.html}`, real
`results/`, `test_shoshin.py`, `make repro`, CI, `NOTICE.md`, `docs/REPORT.md`, MIT `LICENSE`; a 1080p MP4 of the
real dashboard (magenta glow, ledger, compute meter, arrow-key segment), a B&W README GIF from the same run, and a
shot log. Suggested 30 s beats: 0-3 s board flares on an unseen game; 3-12 s first rules learned; 12-20 s compute
meter vs full recipe; 20-27 s a human takes the keys; end card "shoshin. it has never seen this game." The launch
film itself (Remotion, HyperFrames, CPU voiceover) is cut later by Content; nothing is posted without Sebastian's OK.

## Standing rules

Real data only; mock fixtures only under `tests/mock_data/`, never in results, README numbers or the demo. CPU only.
Ponytail (`PONYTAIL.md`). MIT + `NOTICE.md`. Black-and-white athanor README (`readme-ref/README-FORMAT.md`). Tests +
CI. Open a PR; do not merge. Never commit secrets.
