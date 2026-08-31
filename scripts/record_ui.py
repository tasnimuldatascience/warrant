"""Record the browser segment with Playwright.

Playwright records the *page*, not the desktop. That removes the two things wrong with a
screen capture of a browser: it cannot film your open tabs, your bookmarks or your wallpaper,
and it does not depend on where a window happened to sit. The viewport is fixed, so the
recording is the same every time it is made.

    make serve                      # and ask one question, so the model is warm
    python scripts/record_ui.py

The page drives itself via `#/ask/play` -- the same API a visitor calls, waited on for real.
Nothing here is staged: if the system is slow that day, the recording is slow that day.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

BASE = "http://127.0.0.1:8000"
OUT = pathlib.Path("media")
W, H = 1600, 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--seconds", type=float, default=64.0,
                    help="how long to let the sequence run before closing the page")
    ap.add_argument("--out", default=str(OUT / "ui-raw.webm"))
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    tmp = OUT / "_pw"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb"])
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            device_scale_factor=2,          # retina-ish; 1600x1000 encodes to a clean 1080p
            record_video_dir=str(tmp),
            record_video_size={"width": W, "height": H},
            # The recording is the artefact, so a motion-reduced render would flatten the one
            # orchestrated reveal the design has. Ask for the full thing explicitly.
            reduced_motion="no-preference",
            color_scheme="light",
        )
        page = ctx.new_page()
        page.goto(f"{args.base}/#/ask/play", wait_until="networkidle")
        print(f"  playing {args.base}/#/ask/play for {args.seconds:.0f}s", flush=True)

        started = time.perf_counter()
        while time.perf_counter() - started < args.seconds:
            time.sleep(1.0)
            if int(time.perf_counter() - started) % 10 == 0:
                print(f"    {time.perf_counter() - started:.0f}s", flush=True)

        page.close()
        ctx.close()
        browser.close()

    made = sorted(tmp.glob("*.webm"))
    if not made:
        print("no video produced", file=sys.stderr)
        return 1
    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(made[0]), dest)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  wrote {dest} ({dest.stat().st_size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
