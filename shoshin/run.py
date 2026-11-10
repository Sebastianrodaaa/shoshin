"""Replay a held-out game's human recording through WARC under one learning arm and write the per-transition log.

    python -m shoshin.run --games ft09 --arms frozen full gated noreplay random --seeds 0 1 2 --transitions 40
"""
import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import torch

from . import replay
from .gate import Gate, budget, changed_accuracy, rule_key, surprise
from .model import MINIBATCH, STEPS, World, base, checkpoint_for, encode

ARMS = ["frozen", "full", "gated", "noreplay", "random"]
OUT = Path(__file__).resolve().parents[1] / "results" / "runs"


def run(game, arm, seed, transitions, items=None, schedule=None, trace=False, recording=None):
    """One arm on the first `transitions` transitions. `schedule` {t: steps} drives the random-skip control."""
    path = recording or replay.recording(game)
    items = (items or replay.items(path))[:transitions]
    world = World(game, seed, learn=arm != "frozen")
    bank, gate = base.WindowItemReplay(), Gate(world, replay=arm != "noreplay")
    rows = []
    for t, item in enumerate(items):
        started = time.perf_counter()
        prev, real = item[0][-1], item[3]
        pred = world.predict(item)                        # committed before the move is revealed
        wrong, scored = surprise(prev, pred, real)
        n_wrong, n_scored = int(wrong.sum()), int(scored.sum())
        steps = {"frozen": 0, "full": STEPS, "gated": budget(n_wrong, n_scored),
                 "noreplay": budget(n_wrong, n_scored), "random": (schedule or {}).get(t, 0)}[arm]
        if arm == "full":
            bank.add(item)
            world.full_update(bank)
        elif steps:
            gate.learn(item, steps)
        row = dict(t=t, action=list(item[2]), acc=changed_accuracy(prev, pred, real), changed=int((real != prev).sum()),
                   wrong=n_wrong, scored=n_scored, steps=steps, seconds=time.perf_counter() - started,
                   rule=rule_key(item[2], prev, real), level=item[4])
        if trace:
            row.update(prev=encode(prev), pred=encode(pred), real=encode(real))
        rows.append(row)
    assert world.steps == sum(r["steps"] for r in rows)
    meta = dict(game=game, arm=arm, seed=seed, transitions=len(rows), recording=Path(path).name,
                checkpoint=str(checkpoint_for(game).relative_to(checkpoint_for(game).parents[1])),
                minibatch=MINIBATCH, warc_steps=STEPS, torch=torch.__version__, threads=torch.get_num_threads(),
                cpu=platform.processor() or platform.machine(), cores=os.cpu_count(),
                data="ARC-AGI-3 Public Demo human recording (ARC Prize Foundation)")
    return dict(meta=meta, rows=rows)


def random_schedule(gated_rows, seed):
    """Same number of updates and the same step counts as the gated run, at transitions drawn uniformly."""
    counts = [r["steps"] for r in gated_rows if r["steps"]]
    rng = random.Random(seed + 1)
    rng.shuffle(counts)
    return dict(zip(rng.sample(range(len(gated_rows)), len(counts)), counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--transitions", type=int, default=40)
    ap.add_argument("--trace", action="store_true", help="also store the frames (for the dashboard)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for game in a.games:
        items = replay.items(replay.recording(game))
        for seed in a.seeds:
            for arm in a.arms:
                if arm == "frozen" and seed != a.seeds[0]:
                    continue                              # no learning, so every seed is the same run
                f = a.out / f"{game}_{arm}_s{seed}.json"
                if f.exists():
                    continue
                sched = None
                if arm == "random":
                    sched = random_schedule(json.loads((a.out / f"{game}_gated_s{seed}.json").read_text())["rows"], seed)
                res = run(game, arm, seed, a.transitions, items=items, schedule=sched, trace=a.trace)
                f.write_text(json.dumps(res))
                rs = res["rows"]
                accs = [r["acc"] for r in rs if r["acc"] is not None]
                print(f"{game} {arm} s{seed}: acc {sum(accs) / len(accs):.3f} steps {sum(r['steps'] for r in rs)} "
                      f"{len(rs) / sum(r['seconds'] for r in rs):.3f} transitions/s", flush=True)


if __name__ == "__main__":
    main()
