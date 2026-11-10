"""The surprise gate: commit a prediction, compare it with reality, and pay for learning only when wrong."""
import math
from collections import Counter, deque

from .model import MINIBATCH

# ponytail: a quarter of WARC's 20 steps, so no transition can cost more than 25% of the every-move recipe.
MAX_STEPS = 5
BUFFER = 64      # earlier surprises kept for replay


def surprise(prev, pred, real):
    """(wrong, scored) boolean maps over the cells that changed in reality or in the prediction."""
    scored = (real != prev) | (pred != prev)
    return (pred != real) & scored, scored


def budget(n_wrong, n_scored):
    """Gradient steps for one transition: none when right, else scaled by the share of changed cells it got wrong."""
    return 0 if n_wrong == 0 else min(MAX_STEPS, math.ceil(MAX_STEPS * n_wrong / n_scored))


class Gate:
    """Surprise-gated learning on a World. `replay=False` is the forgetting ablation."""

    def __init__(self, world, replay=True):
        self.world, self.replay, self.buffer = world, replay, deque(maxlen=BUFFER)

    def learn(self, item, steps):
        """`steps` steps alternating the new transition and a replayed earlier surprise; then remember it."""
        w = self.world
        new, old = w.views([item]), self.buffer if self.replay else ()
        for k in range(steps):
            src = new if k % 2 == 0 or not old else w.views([w.rng.choice(old)])
            w.step([w.rng.choice(src) for _ in range(MINIBATCH)])
        self.buffer.append(item)


def changed_accuracy(prev, pred, real):
    """Accuracy on the cells that changed in reality (None when nothing changed)."""
    ch = real != prev
    return None if not ch.any() else float((pred[ch] == real[ch]).mean())


def rule_key(action, prev, real):
    """Unsupervised proxy for a game rule: the action id plus the colour change it caused most often."""
    ch = real != prev
    if not ch.any():
        return f"{action[0]}:"
    (a, b), _ = Counter(zip(prev[ch].tolist(), real[ch].tolist())).most_common(1)[0]
    return f"{action[0]}:{a}>{b}"


def exposures_to_criterion(accs, thr=0.8):
    """WARC's EtC on one rule: the first exposure (1-based) from which accuracy stays >= thr; None if never."""
    for i in range(len(accs)):
        if all(a >= thr for a in accs[i:]):
            return i + 1
    return None
