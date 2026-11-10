"""Thin CPU wrapper over WARC: the fold checkpoint that held a game out, a prediction, and one gradient step.

Everything model-shaped is WARC's own code (build_from_checkpoint via FinalPredictor, its strict split check, its
Muon + AdamW optimizer, its rotation views and loss). The only change is precision: FP32 everywhere instead of bf16
(autocast and Muon's Newton-Schulz), because bf16 matmuls on a CPU without bf16 instructions are far slower.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WARC = Path(os.environ.get("WARC_DIR", ROOT / "third_party" / "WARC"))
MODELS = Path(os.environ.get("WARC_MODELS", ROOT / "models"))
sys.path[:0] = [str(WARC / "agent"), str(WARC / "agent" / "VARC"), str(WARC / "agent" / "VARC" / "wm_core" / "src")]

from warc_model import varc2_optim  # noqa: E402
from warc_model.final_ttt import FinalPredictor, NEWEST, OLDER, STEPS, VIEW_COUNT, rotation_slot  # noqa: E402
from hist_wm.eval import ttt_eval as base  # noqa: E402


@torch.no_grad()
def _newton_schulz_fp32(G, steps=5):
    """WARC's Muon Newton-Schulz in FP32: its bf16 matmuls take most of an update on a CPU without bf16 units."""
    X = G.float()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = varc2_optim._NS_A * X + (varc2_optim._NS_B * A + varc2_optim._NS_C * (A @ A)) @ X
    return X.T if G.size(0) > G.size(1) else X


varc2_optim.zeropower_via_newtonschulz5 = _newton_schulz_fp32

CPU = torch.device("cpu")
# ponytail: WARC's minibatch is 8; one sample per step keeps the every-move recipe at ~27 s/transition on 8 CPU
# cores instead of ~216 s. Raise it on a bigger box; every arm shares it, so the step accounting stays fair.
MINIBATCH = int(os.environ.get("SHOSHIN_MINIBATCH", "1"))


def checkpoint_for(game):
    """The fold checkpoint that did not train on `game` (models/manifest.json)."""
    fold = json.loads((MODELS / "manifest.json").read_text())["game_to_fold"][game]
    return MODELS / f"fold{fold}" / "checkpoint_best.pt"


class World:
    """WARC on CPU. `learn=False` is the frozen arm."""

    def __init__(self, game, seed, learn=True):
        torch.manual_seed(seed)
        # strict_split=True: refuses a game the checkpoint was pretrained on.
        self.p = FinalPredictor(str(checkpoint_for(game)), game=game, arm="ttt" if learn else "frozen",
                                device="cpu", seed=seed)
        self.rng = self.p.rng
        self.steps = 0

    @torch.no_grad()
    def predict(self, item):
        """Committed next frame (64, 64) uint8 for a WARC window item (frames, inter, action, next, level)."""
        self.p.model.eval()
        logits, _ = base._forward_dynamics(self.p.model, 3, [item], 0, CPU)
        return logits.argmax(1)[0].to(torch.uint8).numpy()

    def views(self, items):
        """WARC's rotation views of each transition (31 slots per transition, as in final_ttt.bank_update)."""
        data = base.expand_window_items(items, num_color_permute=9, rng=self.rng,
                                        geometric_transforms=base.GEOMETRIC_TRANSFORMS_NO_FLIP)
        return [data[(k // VIEW_COUNT) * VIEW_COUNT + rotation_slot(k % VIEW_COUNT)] for k in range(len(data))]

    def step(self, batch):
        """One gradient step; a non-finite loss or gradient is skipped (WARC's guard) but still counted."""
        p = self.p
        p.model.train()
        p.optimizer.zero_grad(set_to_none=True)
        logits, target = base._forward_dynamics(p.model, 3, batch, 0, CPU)
        loss = base.ffp_loss(logits.float(), target)
        if math.isfinite(float(loss)):
            loss.backward()
            if math.isfinite(float(torch.nn.utils.clip_grad_norm_(p.trainable, 1.))):
                p.optimizer.step()
        p.optimizer.zero_grad(set_to_none=True)
        p.model.eval()
        self.steps += 1

    def full_update(self, bank):
        """WARC's every-move recipe: STEPS steps on the newest 10 + 20 older transitions of `bank`, rotation views."""
        chunk = min(NEWEST, len(bank))
        raw = bank.tail(chunk) + bank.sample(int(OLDER / NEWEST * chunk), exclude_tail=chunk, rng=self.rng)
        data, done = self.views(raw), 0
        while done < STEPS:
            self.rng.shuffle(data)
            for i in range(0, len(data), MINIBATCH):
                if done >= STEPS:
                    break
                self.step(data[i:i + MINIBATCH])
                done += 1
        return STEPS


def decode(s):
    """Hex string of 4096 cells -> (64, 64) uint8."""
    return np.frombuffer(bytes.fromhex("".join("0" + c for c in s)), np.uint8).reshape(64, 64)


def encode(frame):
    return "".join("0123456789abcdef"[v] for v in np.asarray(frame, np.uint8).ravel())
