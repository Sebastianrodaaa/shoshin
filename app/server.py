"""Local dashboard: replays a real shoshin run and lets you play a held-out public game against the model.

    python app/server.py            # http://127.0.0.1:8765
"""
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoshin.gate import Gate, budget, changed_accuracy, surprise  # noqa: E402
from shoshin.model import STEPS, World, encode  # noqa: E402
from hist_window.src.history import HistoryWindow  # noqa: E402

DEMO = ROOT / "results" / "demo"
GAMES = Path(os.environ.get("SHOSHIN_GAMES", ROOT / "games"))   # WARC's warc_eval prepare output
PLAY_GAME = os.environ.get("SHOSHIN_PLAY", "g50t")


class Play:
    """One public game running locally (arc-agi OFFLINE mode), with the fold model learning from your moves."""

    def __init__(self, game):
        os.environ["OPERATION_MODE"] = "OFFLINE"
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction
        self.GameAction = GameAction
        game_id = next(g["game_id"] for g in json.loads((GAMES / "manifest.json").read_text())["games"]
                       if g["game_id"].split("-")[0] == game)
        self.env = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(GAMES / "games")).make(
            game_id, seed=0, save_recording=False)
        self.world, self.game = World(game, seed=0), game
        self.gate = Gate(self.world)
        self.lock = threading.Lock()   # held while the model learns from the last move
        self.moves = 0
        self._observe(self.env.observation_space)
        self.hist = HistoryWindow(3)
        self.hist.reset(self.frame)

    def _observe(self, obs):
        self.frame = np.array(obs.frame[-1], np.uint8)
        self.level = int(obs.levels_completed)
        self.available = [int(a) for a in obs.available_actions]
        self.state = obs.state.name

    def state_dict(self, **extra):
        return {**dict(game=self.game, frame=encode(self.frame), available=self.available, state=self.state,
                       level=self.level, moves=self.moves, steps=self.world.steps, full_steps=STEPS * self.moves), **extra}

    def move(self, kind, x, y):
        if kind not in self.available + [0] or (kind == 6 and not (0 <= x < 64 and 0 <= y < 64)):
            raise ValueError("unavailable action")
        self.lock.acquire()
        learning = False   # once the learner thread owns the lock it releases it; every other exit releases here
        try:
            if kind == 0 or self.state in ("GAME_OVER", "WIN"):
                self._observe(self.env.reset())
                self.hist.reset(self.frame)
                return self.state_dict(reset=True)
            action, prev = (kind, x, y) if kind == 6 else (kind, -1, -1), self.frame
            pred = self.world.predict(self.hist.window_item(action, prev, self.level))  # committed first
            self._observe(self.env.step(self.GameAction.from_id(kind), data={"x": x, "y": y} if kind == 6 else None))
            item = self.hist.window_item(action, self.frame, self.level)
            self.hist.push(action, self.frame)
            self.moves += 1
            wrong, scored = surprise(prev, pred, self.frame)
            steps = budget(int(wrong.sum()), int(scored.sum()))
            out = self.state_dict(prev=encode(prev), pred=encode(pred), wrong=int(wrong.sum()), scored=int(scored.sum()),
                                  acc=changed_accuracy(prev, pred, self.frame), update=steps,
                                  steps=self.world.steps + steps)
            threading.Thread(target=self._learn, args=(item, steps), daemon=True).start()
            learning = True
            return out
        finally:
            if not learning:
                self.lock.release()

    def _learn(self, item, steps):
        try:
            self.gate.learn(item, steps)
        finally:
            self.lock.release()


play = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT / "app"), **k)

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/demo":
            files = sorted(DEMO.glob("*.json"))
            if not files:
                return self.send_json({"error": "no demo run: make demo"}, 404)
            return self.send_json(json.loads(files[0].read_text()))
        if self.path == "/api/summary":
            f = ROOT / "results" / "summary.json"
            return self.send_json(json.loads(f.read_text()) if f.exists() else {})
        return super().do_GET()

    def do_POST(self):
        global play
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 4096)) or b"{}")
            if self.path == "/api/play":
                play = Play(PLAY_GAME)
                return self.send_json(play.state_dict())
            if self.path == "/api/move" and play:
                kind, x, y = (int(body.get(k, -1)) for k in ("kind", "x", "y"))
                return self.send_json(play.move(kind, x, y))
            return self.send_json({"error": "start play first"}, 400)
        except (ValueError, KeyError, StopIteration, FileNotFoundError) as e:
            return self.send_json({"error": str(e)}, 400)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    print(f"shoshin on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
