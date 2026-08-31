"""Thumbnail for the demo video.

1280x720, the repository's own palette. Built rather than screenshotted so the type is set at
thumbnail scale: a downscaled screenshot of a UI is unreadable at the size YouTube actually
shows it, which is why most engineering thumbnails are.
"""
import pathlib

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
PAPER, INK, MUTED = (236, 238, 239), (18, 22, 26), (90, 101, 109)
ACCENT, STAMP, RULE = (26, 79, 122), (155, 44, 34), (167, 177, 183)

FONTS = [
    r"C:\Windows\Fonts\georgia.ttf",
    r"C:\Windows\Fonts\georgiab.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\consolab.ttf",
]


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


serif = lambda s: font(FONTS[0], s)      # noqa: E731
serif_b = lambda s: font(FONTS[1], s)    # noqa: E731
mono = lambda s: font(FONTS[2], s)       # noqa: E731
mono_b = lambda s: font(FONTS[3], s)     # noqa: E731

img = Image.new("RGB", (W, H), PAPER)
d = ImageDraw.Draw(img)

# masthead
mark = serif_b(34)
d.text((64, 46), "WARRANT", font=mark, fill=INK)
# Measured, not guessed: a hardcoded offset put the strapline through the wordmark.
mark_w = d.textlength("WARRANT", font=mark)
d.text((64 + mark_w + 26, 58), "5 CFR  ·  POINT-IN-TIME", font=mono(15), fill=MUTED)
d.line([(64, 104), (W - 64, 104)], fill=INK, width=3)
d.line([(64, 112), (W - 64, 112)], fill=INK, width=1)

# the claim
d.text((64, 158), "ONE QUESTION  ·  TWO DATES", font=mono_b(17), fill=STAMP)
d.text((64, 200), "Two correct", font=serif(88), fill=INK)
d.text((64, 292), "answers.", font=serif(88), fill=INK)

# the split, echoing the Ask screen
y = 432
d.line([(64, y), (W - 64, y)], fill=RULE, width=1)
d.line([(W // 2, y), (W // 2, H - 96)], fill=INK, width=3)

for x, date, txt in ((64, "2024-06-01", "not less than 6 months\nand not more than 4 years"),
                     (W // 2 + 40, "2026-08-26", "up to 4 years\nthe floor was removed")):
    d.text((x, y + 30), "AS OF", font=mono(14), fill=MUTED)
    d.text((x + 62, y + 26), date, font=mono_b(26), fill=INK)
    for i, line in enumerate(txt.split("\n")):
        d.text((x, y + 82 + i * 38), line, font=serif(31), fill=INK)

# footer
d.line([(64, H - 78), (W - 64, H - 78)], fill=INK, width=2)
d.text((64, H - 58), "and it tells you which stage is answerable when it's wrong",
       font=serif(25), fill=MUTED)

out = pathlib.Path(r"D:\Projects\warrant\media\thumbnail.png")
img.save(out, optimize=True)
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {W}x{H})")
