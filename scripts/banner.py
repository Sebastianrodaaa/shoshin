"""docs/screenshots/banner.png: 1280x400, pure black and white. Dürer's Melencolia I (1514) dithered at both edges.

    python scripts/banner.py path/to/melencolia.jpg
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "scripts" / "assets"
W, H = 1280, 400


def edge(src, width, flip):
    """A dithered strip of the engraving that fades into black towards the centre."""
    img = Image.open(src).convert("L")
    img = img.crop((img.width // 12, img.height // 12, img.width * 11 // 12, img.height * 11 // 12))  # skip margins
    img = img.resize((round(img.width * H / img.height), H), Image.LANCZOS)
    img = img.crop((0, 0, width, H)) if not flip else img.crop((img.width - width, 0, img.width, H))
    img = Image.eval(img, lambda v: max(0, min(255, (180 - v) * 2)))   # ink lines become white on black
    fade = Image.linear_gradient("L").rotate(-90 if not flip else 90).resize((width, H))
    img = Image.composite(img, Image.new("L", (width, H), 0), fade)
    return img.convert("1", dither=Image.FLOYDSTEINBERG)


def pixel_text(text, font, size, scale):
    """Render without antialiasing at a small size, then scale up with hard pixels: a bitmap face."""
    f = ImageFont.truetype(str(font), size)
    box = f.getbbox(text)
    img = Image.new("1", (box[2] - box[0] + 4, box[3] - box[1] + 4), 0)
    ImageDraw.Draw(img).text((2 - box[0], 2 - box[1]), text, font=f, fill=1)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def main(src):
    out = Image.new("1", (W, H), 0)
    out.paste(edge(src, 360, False), (0, 0))
    out.paste(edge(src, 360, True), (W - 360, 0))
    title = pixel_text("shoshin", FONTS / "unifraktur.ttf", 44, 4)
    out.paste(title, ((W - title.width) // 2, 92))
    tag = pixel_text("it has never seen this game. watch it learn.", FONTS / "jbm.ttf", 14, 2)
    out.paste(tag, ((W - tag.width) // 2, 92 + title.height + 28))
    path = ROOT / "docs" / "screenshots" / "banner.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("L").save(path)
    print(path)


if __name__ == "__main__":
    main(sys.argv[1])
