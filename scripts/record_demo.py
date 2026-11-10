"""Record the real dashboard: a 1080p MP4 (docs/media/shoshin-demo.mp4), a black-and-white README GIF cut from it
(docs/screenshots/demo.gif), B&W stills, and a shot log (docs/media/SHOTLOG.md).

    python scripts/record_demo.py        # starts app/server.py itself
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
MEDIA, SHOTS = ROOT / "docs" / "media", ROOT / "docs" / "screenshots"
PORT = 8777
END_CARD = ("document.body.insertAdjacentHTML('beforeend', `<div style=\"position:fixed;inset:0;background:#f4eee2;"
            "display:flex;align-items:center;justify-content:center;font:64px Silkscreen;color:#141210\">"
            "shoshin. it has never seen this game.</div>`)")


def bw(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "format=gray", str(dst)], check=True)


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, str(ROOT / "app" / "server.py")], env={**__import__("os").environ,
                              "PORT": str(PORT)})
    log, t0 = [], None
    try:
        time.sleep(4)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(MEDIA / "raw"),
                                      record_video_size={"width": 1920, "height": 1080})
            page = ctx.new_page()

            def mark(text):
                log.append((time.time() - t0, text))

            t0 = time.time()
            page.goto(f"http://127.0.0.1:{PORT}/?ms=450")
            mark("unseen game: the first committed guess is wrong, the surprise map flares")
            page.wait_for_timeout(2500)
            page.screenshot(path=str(MEDIA / "flare.png"))
            mark("first rules learned; ledger ticks; surprise patches cool to lilac")
            page.wait_for_timeout(9000)
            page.screenshot(path=str(MEDIA / "ledger.png"))
            mark("compute meter: gated gradient steps vs the full every-move WARC recipe")
            page.wait_for_timeout(8000)
            page.screenshot(path=str(MEDIA / "meter.png"))
            mark("play yourself: a human takes the arrow keys and tries to surprise it")
            page.click("#btn-play")
            page.wait_for_function("document.getElementById('caption').textContent.startsWith('your move')", timeout=120000)
            for key in ["ArrowRight", "ArrowRight", "ArrowDown", "ArrowDown", "ArrowLeft", "ArrowUp", "ArrowRight",
                        "ArrowDown"]:
                page.keyboard.press(key)
                page.wait_for_timeout(900)
            page.screenshot(path=str(MEDIA / "play.png"))
            mark("end card")
            page.evaluate(END_CARD)
            page.wait_for_timeout(3000)
            mark("end")
            video = page.video.path()
            ctx.close()
            browser.close()
    finally:
        server.terminate()
    mp4 = MEDIA / "shoshin-demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "scale=1920:1080", "-r", "30", "-crf", "20", str(mp4)], check=True)
    shutil.rmtree(MEDIA / "raw", ignore_errors=True)
    gif = SHOTS / "demo.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0", "-t", "20", "-i", str(mp4), "-vf",
                    "fps=8,scale=960:-1:flags=lanczos,format=gray,split[a][b];[a]palettegen=max_colors=16[p];"
                    "[b][p]paletteuse=dither=bayer", str(gif)], check=True)
    for name in ("flare", "ledger", "meter", "play"):
        bw(MEDIA / f"{name}.png", SHOTS / f"{name}.png")
    lines = ["# Shot log (real run, recorded by scripts/record_demo.py)", "",
             "Source: `shoshin-demo.mp4`, 1920x1080, 30 fps, no audio. Captions are on screen.", "",
             "| from (s) | to (s) | shot |", "| --- | --- | --- |"]
    for (a, text), (b, _) in zip(log, log[1:]):
        lines.append(f"| {a:.1f} | {b:.1f} | {text} |")
    (MEDIA / "SHOTLOG.md").write_text("\n".join(lines) + "\n")
    print(mp4, gif)


if __name__ == "__main__":
    main()
