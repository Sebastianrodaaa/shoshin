"""Smoke test on a real held-out recording: ft09, replayed through fold3 (which never trained on ft09), on CPU.

    python3 test_shoshin.py
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from shoshin import replay  # noqa: E402
from shoshin.gate import Gate, budget, surprise  # noqa: E402
from shoshin.model import STEPS, FinalPredictor, World, checkpoint_for  # noqa: E402
from shoshin.run import random_schedule, run  # noqa: E402

GAME, T = "ft09", 8
MOCK = ROOT / "tests" / "mock_data" / "corrupted_ft09.json"


class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = replay.items(replay.recording(GAME))
        cls.gated = run(GAME, "gated", 0, T, items=cls.items)["rows"]

    def test_held_out_fold(self):
        manifest = json.loads((checkpoint_for(GAME).parents[1] / "manifest.json").read_text())
        self.assertEqual(checkpoint_for(GAME).parent.name, "fold3")
        self.assertIn(GAME, json.dumps(manifest["game_to_fold"]))
        with self.assertRaises(Exception):     # WARC's strict split: fold3 trained on bp35, so it refuses it
            FinalPredictor(str(checkpoint_for(GAME)), game="bp35", arm="frozen", device="cpu", seed=0)

    def test_gate_skips_predicted_moves(self):
        for r in self.gated:
            self.assertEqual(r["steps"], budget(r["wrong"], r["scored"]))
            if r["wrong"] == 0:
                self.assertEqual(r["steps"], 0)
        self.assertTrue(any(r["wrong"] == 0 for r in self.gated), "expected at least one predicted move")

    def test_corrupted_frame_triggers_update(self):
        """tests/mock_data: a real transition the model predicts, with its next frame corrupted (labelled mock)."""
        fx = json.loads(MOCK.read_text())
        self.assertTrue(fx["mock"])
        item = self.items[fx["t"]]
        w = World(GAME, 0)
        pred = w.predict(item)
        real = item[3].copy()
        y0, x0, n = fx["block"]
        real[y0:y0 + n, x0:x0 + n] = (real[y0:y0 + n, x0:x0 + n] + 1) % 16
        wrong, scored = surprise(item[0][-1], pred, real)
        steps = budget(int(wrong.sum()), int(scored.sum()))
        self.assertGreater(steps, 0)
        corrupted = (item[0], item[1], item[2], real, item[4])
        Gate(w).learn(corrupted, steps)
        self.assertEqual(w.steps, steps)

    def test_fewer_steps_than_full(self):
        full = run(GAME, "full", 0, 2, items=self.items)["rows"]
        self.assertEqual(sum(r["steps"] for r in full), 2 * STEPS)
        self.assertLess(sum(r["steps"] for r in self.gated[:2]), 2 * STEPS)
        sched = random_schedule(self.gated, 0)
        self.assertEqual(sorted(sched.values()), sorted(r["steps"] for r in self.gated if r["steps"]))

    def test_results_regenerate_readme(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import plots
        runs = ROOT / "results" / "runs"
        if not any(runs.glob("*.json")):
            self.skipTest("no cached results")
        s, _ = plots.summarise(runs)
        if not s["games"]:
            self.skipTest("no game with every arm cached yet")
        text = plots.table(s)
        for f in ("README.md", "docs/REPORT.md"):
            self.assertIn(text, (ROOT / f).read_text(), f"{f} tables are stale: run make repro")
        self.assertTrue(np.isfinite(s["arms"]["gated"]["acc"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
