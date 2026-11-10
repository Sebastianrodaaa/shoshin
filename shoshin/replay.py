"""Offline replay of the public ARC-AGI-3 human recordings, checked against WARC's data/SHA256SUMS."""
import hashlib
from pathlib import Path

from .model import ROOT  # noqa: F401  (sets up WARC's import path)
from varc2.data import parse_recording  # noqa: E402
from hist_wm.src.hist_data import steps_to_window_items  # noqa: E402

DATA = ROOT / "data"


def recording(game, index=0):
    """The `index`-th recording of `game` by file name, as WARC lists it; refuses a file whose SHA-256 differs."""
    sums = [line.split() for line in (DATA / "SHA256SUMS").read_text().splitlines() if f"/{game}/" in line]
    digest, rel = sorted(sums, key=lambda x: Path(x[1]).name)[index]
    path = DATA / rel
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: see data/README.md for the download")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{path} does not match data/SHA256SUMS")
    return path


def items(path, hist_n=3):
    """WARC window items (frames, inter, action, next, level), one per human transition, in play order."""
    return steps_to_window_items(parse_recording(Path(path)), hist_n)
