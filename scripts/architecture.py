"""Record `media/architecture.html` into the feed assets: a looping GIF and an mp4.

The diagram is static except for its connectors, so the only thing being captured is the
flow. That is deliberate -- a feed renders this at roughly 500px wide, and boxes that fade
in one at a time are unreadable at that size while a moving line is not.

    python scripts/architecture.py

Fonts are fetched over the network and Chromium does not paint until they land, so the
page does not start its own loop: the recorder waits for the fonts, lets the capture
settle, and only then calls `__run()`. Without that the first cycle of the loop is missing
from the recording, which is the same trap `scripts/film.py` documents.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import time

MEDIA = pathlib.Path("media")
W, H = 1700, 1060

SETTLE = 1.2      #: recorder warm-up before the loop starts
RECORD = 15.0     #: one full cycle is ~12.5s; the surplus is trimmed to the loop point
GIF_W = 1400      #: the label size a reader gets is font-size x (GIF_W / canvas)
GIF_FPS = 10      #: 10, not 7: fewer frames means bigger deltas between them, and GIF
                  #: encodes deltas -- dropping to 7 fps made the file larger, not smaller


def record(dest: pathlib.Path, still: pathlib.Path | None) -> float:
    from playwright.sync_api import sync_playwright

    tmp = MEDIA / "_arch"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--force-color-profile=srgb"])
        ctx = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=2,
                            record_video_dir=str(tmp),
                            record_video_size={"width": W, "height": H},
                            reduced_motion="no-preference")
        pg = ctx.new_page()
        pg.goto((MEDIA / "architecture.html").resolve().as_uri())
        pg.wait_for_function("document.fonts.status === 'loaded'", timeout=30_000)
        time.sleep(SETTLE)
        pg.evaluate("window.__run()")
        time.sleep(RECORD)
        cycle_ms = pg.evaluate("window.__cycleMs || 0")
        # The still is taken off the page, not off a frame of the video: the mp4 is
        # 4:2:0, and chroma subsampling is unkind to 1.5px rules on white.
        if still is not None:
            pg.evaluate("window.__freeze()")
            time.sleep(0.4)
            pg.screenshot(path=str(still))
        pg.close()
        ctx.close()
        b.close()
    raw = sorted(tmp.glob("*.webm"))[0]
    shutil.move(str(raw), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    return float(cycle_ms) / 1000.0


def main() -> int:
    webm = MEDIA / "_arch.webm"
    cycle = record(webm, still=MEDIA / "architecture.png")
    print(f"  wrote architecture.png  ({(MEDIA / 'architecture.png').stat().st_size / 1e3:.0f} KB)")
    print(f"  recorded {webm.name}  --  one cycle is {cycle:.2f}s")

    # Trim the warm-up. The loop resets its wires and holds, so starting the clip a beat
    # after the settle lands on an empty canvas -- which is where the loop returns to.
    start = f"{SETTLE:.2f}"
    span = ["-t", f"{cycle:.3f}"] if cycle > 1 else []

    mp4 = MEDIA / "warrant-architecture.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", start, *span, "-i", str(webm),
         "-vf", "scale=-2:1080:flags=lanczos,pad=1920:1080:(ow-iw)/2:0:color=white",
         "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
         "-an", "-movflags", "+faststart", str(mp4)], check=True)
    print(f"  wrote {mp4.name}  ({mp4.stat().st_size / 1e6:.1f} MB)")

    # Two passes: a palette built from the whole clip, then the map. One pass dithers the
    # thin rules into mush -- this diagram is mostly 1.5px lines on white.
    pal = MEDIA / "_arch-palette.png"
    chain = f"fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", start, *span, "-i", str(webm),
                    "-vf", f"{chain},palettegen=max_colors=64:stats_mode=diff",
                    str(pal)], check=True)
    gif = MEDIA / "warrant-architecture.gif"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", start, *span, "-i", str(webm),
                    "-i", str(pal), "-lavfi",
                    f"{chain}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", str(gif)], check=True)
    pal.unlink(missing_ok=True)
    print(f"  wrote {gif.name}  ({gif.stat().st_size / 1e6:.1f} MB)")

    webm.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
