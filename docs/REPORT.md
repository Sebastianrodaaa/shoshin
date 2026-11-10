# shoshin: surprise-gated test-time learning for WARC on unseen ARC-AGI-3 games

Small-scale, CPU-only study. Every number below is regenerated from `results/` by `make repro`
(`scripts/plots.py`). Nothing here is mocked; the only mock data in the repository is the corrupted-frame test fixture
in `tests/mock_data/`, which never feeds a result.

## Question

WARC's test-time recipe takes 20 gradient steps after every transition, whether or not the model already predicted
it. On a CPU that is the whole budget. Can the model learn only when it is surprised, and keep most of the accuracy?

Hypothesis (fixed before the run, from the brief). Replaying public human recordings of each held-out game through
the WARC fold checkpoint that did not train on it, surprise-gated learning

1. keeps changed-cell accuracy within 2 points absolute of the full every-move recipe,
2. while spending at most 25% of its gradient steps,
3. beats a random-skip control that gets the same number of updates, and
4. runs at 1 or more transitions per second on this CPU.

It fails if (1) or (3) does not hold over 3 seeds (95% bootstrap CIs).

## Method

**Model.** WARC (Onoda, Kawano, Takashiro, Hayashi; arXiv:2611.12324), the released fold checkpoints, loaded on CPU
with WARC's own `FinalPredictor` (`shoshin/model.py`). Each game is only shown to the fold that held it out
(`models/manifest.json`); WARC's strict split refuses the others, and `test_shoshin.py` checks that it does.

**Data.** The ARC Prize Foundation's ARC-AGI-3 Public Demo human recordings, checked against WARC's
`data/SHA256SUMS` (340 of 340 listed files verified locally), parsed with WARC's parser into WARC's window items.
Each run replays the first T transitions of the first recording (by file name) of a game.

**Commit, compare, learn** (`shoshin/gate.py`). Before each transition the model predicts the next frame. Scored
cells are the cells that changed in reality or in the prediction. Wrong cells are scored cells where the guess
differs from reality. Changed-cell accuracy is 1 minus wrong/scored (undefined when nothing is scored). The update
budget is `ceil(5 * wrong / scored)` steps, zero when nothing is wrong; 5 is a quarter of WARC's 20, so a single
transition can never cost more than 25% of the every-move recipe. Each step trains on the new transition plus a
replay buffer of up to 64 earlier surprises, using WARC's loss, augmentation views and Muon + AdamW optimiser.

**Arms** (`shoshin/run.py`). frozen; full WARC (20 steps after every transition, WARC's replay); surprise-gated;
surprise-gated without the surprise buffer (forgetting ablation); random-skip, which gets exactly the gated run's
list of step counts (same seed), placed at uniformly random transitions.

**CPU changes, applied to every arm.** One sample per step instead of WARC's 8 (`ponytail:` note in
`shoshin/model.py`): at 8 the every-move recipe is about 216 s per transition on this box. WARC's Muon
Newton-Schulz iteration runs in fp32 instead of bf16, because this CPU has no bf16 units. The step accounting is the
same for every arm, so the comparison stays fair; absolute accuracy is not WARC's GPU number.

**Metrics.** Changed-cell accuracy per transition, averaged per run, then over games. 95% CIs are a bootstrap over
transitions and seeds (2000 resamples). Exposures to criterion (EtC) is an unsupervised proxy: a "rule" is an
action id plus the colour change it caused most often; EtC is the first exposure from which that rule stays at 80% or
better. It is a proxy, not WARC's MechEval (the annotations were not released as of 2026-11-10).

**Hardware.** One VM, 8 vCPU (Intel Xeon, no AVX-512 bf16), torch 2.7.0+cpu, no GPU.

## Results

<!-- results:start -->
| arm | changed-cell accuracy % (95% CI) | gradient steps | share of full | transitions/s | EtC proxy |
| --- | --- | --- | --- | --- | --- |
| frozen (no learning) | 27.6 (23.0 to 32.3) | 0 | 0.0% | 2.39 | 1.8 (10.0% reach) |
| full WARC (20 steps every move) | 61.1 (54.1 to 67.5) | 2880 | 100.0% | 0.01 | 3.5 (48.7% reach) |
| surprise-gated | 53.9 (47.5 to 60.3) | 410 | 14.2% | 0.09 | 3.4 (44.8% reach) |
| surprise-gated, no replay | 53.3 (46.9 to 59.4) | 416 | 14.5% | 0.08 | 5.1 (37.6% reach) |
| random-skip, matched updates | 49.3 (42.7 to 56.1) | 410 | 14.2% | 0.09 | 4.8 (44.3% reach) |

| comparison | difference, points (95% CI) |
| --- | --- |
| gated minus full | -7.2 (-11.4 to -3.4) |
| gated minus random | +4.7 (+1.6 to +8.1) |
| gated minus noreplay | +0.6 (-2.1 to +3.6) |
| full minus frozen | +33.5 (+27.6 to +39.2) |

| hypothesis part | holds |
| --- | --- |
| 1 accuracy within 2 points of full (CI lower bound > -0.02) | no |
| 2 at most 25% of full gradient steps | yes |
| 3 beats matched random-skip (CI lower bound > 0) | yes |
| 4 at least 1 transition per second | no |
| holds | no |
<!-- results:end -->

Per game (changed-cell accuracy %, mean over seeds, and gradient steps):

<!-- pergame:start -->
| game | fold | frozen (no learning) | full WARC (20 steps every move) | surprise-gated | surprise-gated, no replay | random-skip, matched updates |
| --- | --- | --- | --- | --- | --- | --- |
| bp35 | fold1 | 3.6 (0 steps) | 66.9 (480 steps) | 51.4 (76 steps) | 51.3 (76 steps) | 49.2 (76 steps) |
| cn04 | fold2 | 56.2 (0 steps) | 70.8 (480 steps) | 70.8 (50 steps) | 71.3 (52 steps) | 68.4 (50 steps) |
| ft09 | fold3 | 0.0 (0 steps) | 57.2 (480 steps) | 51.0 (62 steps) | 49.6 (64 steps) | 36.9 (62 steps) |
| g50t | fold3 | 60.3 (0 steps) | 68.4 (480 steps) | 68.7 (52 steps) | 60.8 (60 steps) | 64.6 (52 steps) |
| m0r0 | fold2 | 45.6 (0 steps) | 67.2 (480 steps) | 65.9 (59 steps) | 66.8 (56 steps) | 59.8 (59 steps) |
| r11l | fold1 | 0.0 (0 steps) | 36.1 (480 steps) | 15.7 (110 steps) | 20.1 (108 steps) | 16.7 (110 steps) |
<!-- pergame:end -->

**Wall-clock.** The evaluation ran three processes side by side, two threads each, so its per-transition times are
slower than one process on the whole machine. Hypothesis part 4 is judged on the dedicated smoke run (ft09, 20
transitions, seed 0, one process, 8 threads):

<!-- timing:start -->
| arm | seconds per transition | transitions/s | gradient steps |
| --- | --- | --- | --- |
| frozen (no learning) | 0.29 | 3.496 | 0 |
| full WARC (20 steps every move) | 43.94 | 0.023 | 400 |
| surprise-gated | 5.18 | 0.193 | 56 |
| surprise-gated, no replay | 5.73 | 0.174 | 54 |
| random-skip, matched updates | 5.80 | 0.172 | 56 |
<!-- timing:end -->

![accuracy and compute](screenshots/results.png)

## Verdict

**The hypothesis fails.** Over six held-out games, 24 transitions each, and 3 seeds:

1. Accuracy: surprise-gated is 7.2 points behind the full every-move recipe (95% CI -11.4 to -3.4). The bar was
   "within 2 points". **Fails.**
2. Compute: it spends 14.2% of the full recipe's gradient steps (410 vs 2880). The bar was "at most 25%". **Holds.**
3. Control: it beats random-skip with the identical step list by 4.7 points (95% CI +1.6 to +8.1). **Holds.**
4. Speed: about 0.2 transitions per second on this 8-vCPU box, against a bar of 1 (see the timing table). One
   batch-1 WARC update costs about 1.5 s here, so any update at all already misses the bar. **Fails.**

The gate is not the problem: it learns from the right transitions, because the same budget spent at random does
worse, and the few steps it spends buy most of the full recipe's gain over frozen (+26.3 of +33.5 points). The cost is
the 85% of steps it skips: on these recordings, the extra steps on already-predicted moves are not wasted. The gap
sits mostly in two games, r11l (-20.4) and bp35 (-15.5). On cn04, g50t and m0r0 gated matches full within about one
point. The replay buffer did not measurably help over 24 transitions (+0.6, CI -2.1 to +3.6); forgetting needs
longer runs to show up.

The every-move recipe is not cheap on CPU either (about 0.03 transitions per second), so gating is still the only
arm of the two that is anywhere near interactive. It is short of the bar.

## Related work

- **WARC** (arXiv:2611.12324): the model, the checkpoints, the held-out folds and the every-move test-time recipe
  that shoshin gates. shoshin changes when and how much it learns, not what.
- **EATA** (Niu et al., ICML 2022, "Efficient Test-Time Model Adaptation without Forgetting"): skips test samples that
  are unreliable or redundant and adds an anti-forgetting term. shoshin's gate is the world-model analogue, with a
  better signal than entropy: the next frame arrives, so the error is observed, not estimated.
- **Tent** (Wang et al., ICLR 2021) and test-time training (Sun et al., ICML 2020): adapt at test time on every batch.
- **Experience replay** for continual learning (Rolnick et al., NeurIPS 2019): the surprise buffer is a tiny version.
- **Prioritised experience replay** (Schaul et al., ICLR 2016): spend updates where the error is large.
- **Surprise and curiosity** (Schmidhuber 1991; Pathak et al., ICML 2017): prediction error as the learning signal.
- **ARC-AGI-3** (ARC Prize Foundation): the public games and the Public Demo human recordings.

## Limitations

- Small scale: six held-out games (two per fold), the first 24 transitions of one recording each, 3 seeds. The
  CIs are wide and the result may not hold over whole recordings.
- One sample per step and fp32 Muon change WARC's recipe on CPU; the full arm is "WARC's schedule under the same
  CPU settings", not WARC's published GPU numbers.
- The rule proxy groups by action and dominant colour change. It over-splits some mechanics and merges others.
- Bootstrap over transitions treats transitions as independent; they are not.
- Timings are from one VM.
