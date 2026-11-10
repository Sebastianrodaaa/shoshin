"""Summarise results/runs/*.json: bootstrap CIs, hypothesis verdict, README/REPORT tables, black-and-white plots.

    python scripts/plots.py [results/runs]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shoshin.gate import exposures_to_criterion  # noqa: E402

ARMS = ["frozen", "full", "gated", "noreplay", "random"]
NAMES = {"frozen": "frozen (no learning)", "full": "full WARC (20 steps every move)", "gated": "surprise-gated",
         "noreplay": "surprise-gated, no replay", "random": "random-skip, matched updates"}
REPS = 2000
SMOKE = ROOT / "results" / "smoke"
FOLD = {}


def load(runs):
    """{game: {arm: {seed: rows}}}; frozen is seed-independent and stands in for every seed."""
    out = defaultdict(lambda: defaultdict(dict))
    for f in sorted(Path(runs).glob("*.json")):
        d = json.loads(f.read_text())
        out[d["meta"]["game"]][d["meta"]["arm"]][d["meta"]["seed"]] = d["rows"]
        FOLD[d["meta"]["game"]] = d["meta"]["checkpoint"].split("/")[0]
    for arms in out.values():
        seeds = sorted({s for a, v in arms.items() if a != "frozen" for s in v})
        if "frozen" in arms and seeds:
            only = next(iter(arms["frozen"].values()))
            arms["frozen"] = {s: only for s in seeds}
    return out


def acc_matrix(by_seed):
    seeds = sorted(by_seed)
    return np.array([[np.nan if r["acc"] is None else r["acc"] for r in by_seed[s]] for s in seeds])


def boot(mats, rng):
    """Bootstrap the mean over games of (seed mean of transition mean); resamples seeds and transitions per game.
    `mats` is a list (per game) of (seeds, T) arrays, already paired differences when comparing arms."""
    stats = np.empty(REPS)
    for k in range(REPS):
        per_game = []
        for m in mats:
            m = m[:, ~np.isnan(m).any(0)]
            s = rng.integers(0, m.shape[0], m.shape[0])
            t = rng.integers(0, m.shape[1], m.shape[1])
            per_game.append(m[s][:, t].mean())
        stats[k] = np.mean(per_game)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def point(mats):
    return float(np.mean([np.nanmean(m[:, ~np.isnan(m).any(0)]) for m in mats]))


def etc(rows):
    """Exposures-to-criterion proxy: mean over recurring rules (>= 2 exposures) that reach 80%, and reach rate."""
    by = defaultdict(list)
    for r in rows:
        if r["acc"] is not None and r["rule"] and not r["rule"].endswith(":"):
            by[r["rule"]].append(r["acc"])
    hits = [exposures_to_criterion(a) for a in by.values() if len(a) >= 2]
    got = [h for h in hits if h]
    return (float(np.mean(got)) if got else None), (len(got) / len(hits) if hits else None), len(hits)


def summarise(runs):
    data, rng = load(runs), np.random.default_rng(0)
    games = sorted(g for g, a in data.items() if all(x in a for x in ARMS))
    s = dict(games=games, arms={}, per_game={}, comparisons={}, hypothesis={})
    for arm in ARMS:
        mats = [acc_matrix(data[g][arm]) for g in games]
        steps = [np.mean([sum(r["steps"] for r in rows) for rows in data[g][arm].values()]) for g in games]
        secs = [np.mean([sum(r["seconds"] for r in rows) / len(rows) for rows in data[g][arm].values()]) for g in games]
        e = [etc(rows) for g in games for rows in data[g][arm].values()]
        s["arms"][arm] = dict(acc=point(mats), acc_ci=boot(mats, rng), steps=float(np.sum(steps)),
                              seconds_per_transition=float(np.mean(secs)),
                              transitions_per_second=float(1 / np.mean(secs)),
                              etc_mean=_mean([x[0] for x in e]), etc_reach=_mean([x[1] for x in e]))
    for g in games:
        s["per_game"][g] = {"fold": FOLD[g]} | {arm: dict(acc=point([acc_matrix(data[g][arm])]),
                                      steps=float(np.mean([sum(r["steps"] for r in v) for v in data[g][arm].values()])),
                                      seeds=len(data[g][arm]), transitions=len(next(iter(data[g][arm].values()))))
                           for arm in ARMS}
    for a, b in [("gated", "full"), ("gated", "random"), ("gated", "noreplay"), ("full", "frozen")]:
        diffs = [acc_matrix(data[g][a]) - acc_matrix(data[g][b]) for g in games]
        s["comparisons"][f"{a}-{b}"] = dict(diff=point(diffs), ci=boot(diffs, rng))
    A, C = s["arms"], s["comparisons"]
    s["hypothesis"] = {
        "1 accuracy within 2 points of full (CI lower bound > -0.02)": C["gated-full"]["ci"][0] > -0.02,
        "2 at most 25% of full gradient steps": A["gated"]["steps"] <= 0.25 * A["full"]["steps"],
        "3 beats matched random-skip (CI lower bound > 0)": C["gated-random"]["ci"][0] > 0,
        "4 at least 1 transition per second": A["gated"]["transitions_per_second"] >= 1,
    }
    s["hypothesis"]["holds"] = all(s["hypothesis"].values())
    return s, data


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def table(s):
    A, C, full = s["arms"], s["comparisons"], s["arms"]["full"]["steps"]
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    lines = ["| arm | changed-cell accuracy % (95% CI) | gradient steps | share of full | transitions/s | EtC proxy |",
             "| --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        a = A[arm]
        e = "-" if a["etc_mean"] is None else f"{a['etc_mean']:.1f} ({pct(a['etc_reach'])}% reach)"
        lines.append(f"| {NAMES[arm]} | {pct(a['acc'])} ({pct(a['acc_ci'][0])} to {pct(a['acc_ci'][1])}) | "
                     f"{a['steps']:.0f} | {pct(a['steps'] / full) if full else '-'}% | {a['transitions_per_second']:.2f} | {e} |")
    lines += ["", "| comparison | difference, points (95% CI) |", "| --- | --- |"]
    for k, c in C.items():
        lines.append(f"| {k.replace('-', ' minus ')} | {100 * c['diff']:+.1f} ({100 * c['ci'][0]:+.1f} to {100 * c['ci'][1]:+.1f}) |")
    lines += ["", "| hypothesis part | holds |", "| --- | --- |"]
    lines += [f"| {k} | {'yes' if v else 'no'} |" for k, v in s["hypothesis"].items()]
    return "\n".join(lines)


def per_game(s):
    lines = ["| game | fold | " + " | ".join(NAMES[a] for a in ARMS) + " |", "| --- | --- |" + " --- |" * len(ARMS)]
    for g, v in s["per_game"].items():
        lines.append(f"| {g} | {v['fold']} | " + " | ".join(f"{100 * v[a]['acc']:.1f} ({v[a]['steps']:.0f} steps)"
                                                        for a in ARMS) + " |")
    return "\n".join(lines)


def timing(smoke):
    """Wall-clock from the dedicated smoke run (one process, every core), which hypothesis part 4 is judged on."""
    lines = ["| arm | seconds per transition | transitions/s | gradient steps |", "| --- | --- | --- | --- |"]
    for arm in ARMS:
        f = next(Path(smoke).glob(f"*_{arm}_s0.json"), None)
        if f:
            rows = json.loads(f.read_text())["rows"]
            sec = sum(r["seconds"] for r in rows) / len(rows)
            lines.append(f"| {NAMES[arm]} | {sec:.2f} | {1 / sec:.3f} | {sum(r['steps'] for r in rows)} |")
    return "\n".join(lines)


def splice(path, text, key="results"):
    """Replace the block between <!-- key:start/end --> markers (idempotent, so `make repro` regenerates it)."""
    p = ROOT / path
    if p.exists():
        pat = rf"(<!-- {key}:start -->\n).*?(<!-- {key}:end -->)"
        p.write_text(re.sub(pat, lambda m: m[1] + text.rstrip("\n") + "\n" + m[2], p.read_text(), flags=re.S))


def plot(s, data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                         "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white",
                         "ytick.color": "white", "axes.edgecolor": "#888888", "font.family": "monospace"})
    styles = {"frozen": ("#666666", ":"), "full": ("white", "-"), "gated": ("white", "--"),
              "noreplay": ("#aaaaaa", "-."), "random": ("#888888", (0, (1, 3)))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=110)
    for arm in ARMS:
        c, ls = styles[arm]
        cum = None
        for g in s["games"]:
            run = np.nanmean(acc_matrix(data[g][arm]), 0)
            c_run = np.array([np.nanmean(run[:i]) if np.isfinite(run[:i]).any() else 0 for i in range(1, len(run) + 1)])
            cum = c_run if cum is None else cum[:len(c_run)] + c_run[:len(cum)]
        axes[0].plot(np.arange(1, len(cum) + 1), 100 * cum / len(s["games"]), color=c, ls=ls, lw=1.6, label=NAMES[arm])
        axes[1].bar(NAMES[arm].split(" (")[0].replace(", matched updates", ""), s["arms"][arm]["steps"],
                    color="black", edgecolor=c, hatch={"-": "", "--": "///"}.get(ls, "..") if isinstance(ls, str) else "..")
    axes[0].set(xlabel="transition", ylabel="running changed-cell accuracy %", title="accuracy over the recording")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(ylabel="gradient steps (all games, mean over seeds)", title="compute spent")
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "results.png")
    plt.close(fig)


def main():
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "runs"
    s, data = summarise(runs)
    if not s["games"]:
        sys.exit(f"no complete game in {runs}")
    meta = json.loads(next(Path(runs).glob("*.json")).read_text())["meta"]
    s["setup"] = {k: meta[k] for k in ("transitions", "minibatch", "warc_steps", "torch", "threads", "cores", "data")}
    (ROOT / "results" / "summary.json").write_text(json.dumps(s, indent=1))
    text = table(s)
    for f in ("README.md", "docs/REPORT.md"):
        splice(f, text)
    splice("docs/REPORT.md", per_game(s), "pergame")
    if SMOKE.exists():
        splice("docs/REPORT.md", timing(SMOKE), "timing")
    (ROOT / "docs" / "screenshots").mkdir(parents=True, exist_ok=True)
    plot(s, data, ROOT / "docs" / "screenshots")
    print(text)


if __name__ == "__main__":
    main()
