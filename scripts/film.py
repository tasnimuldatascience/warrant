"""Build the film: scene timings from the real narration, then record and mux.

The order matters. `scripts/voice.py` speaks each line and measures it; this reads those
durations and lays the scenes against them, so a sentence and the thing it describes land
together. Timing visuals to a guessed narration and hoping the voice fits is how a cut ends
up explaining something two beats after it happened.

    python scripts/voice.py     # first: the audio, and its real durations
    python scripts/film.py      # then: the film
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time

MEDIA = pathlib.Path("media")
VOICE = MEDIA / "_voice"
W, H = 1600, 900

#: Silence after a line inside a scene, and the longer breath between scenes. Both are beats
#: a person would take; without them every sentence rear-ends the next.
GAP_LINE, GAP_SCENE = 0.34, 0.86
LEAD_IN = 1.1          #: quiet before the first word, so a viewer arrives before it starts
TAIL = 2.6             #: the close holds, so a loop does not snap away from the URL


#: Caption geometry. Two lines of ~42 characters is the width a player renders without
#: shrinking the type or spilling out of the frame; a cue longer than that is split and the
#: line's measured duration divided between the pieces by character count.
CUE_CHARS, CUE_LINES = 42, 2


def wrap(text: str, width: int = CUE_CHARS) -> list[str]:
    """Greedy word wrap. A word wider than the measure gets its own line rather than a break."""
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def stamp(sec: float) -> str:
    """SRT wants HH:MM:SS,mmm -- a comma before the milliseconds, not a period."""
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(manifest: list[dict], dest: pathlib.Path) -> int:
    """A subtitle sidecar, off the same manifest the scene timings come from.

    Not burned in: every claim in this film is already set as type on screen, so a burned
    caption would print each sentence twice. The sidecar exists because YouTube indexes it,
    and because a viewer who needs captions should not be handed a video that assumes hearing.

    Reads the `caption` column rather than the spoken one. The voice needs numbers spelled
    out, and a caption reading "nine thousand, seven hundred and thirty" beside a graphic
    reading 9,730 looks like a transcription error.
    """
    cues: list[tuple[float, float, str]] = []
    for m in manifest:
        lines = wrap(m.get("caption") or m["text"])
        chunks = [lines[i:i + CUE_LINES] for i in range(0, len(lines), CUE_LINES)]
        weights = [sum(len(x) for x in c) for c in chunks]
        t = m["start"]
        for chunk, w in zip(chunks, weights, strict=True):
            span = m["seconds"] * w / sum(weights)
            cues.append((t, t + span, "\n".join(chunk)))
            t += span

    blocks = [f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n"
              for i, (a, b, text) in enumerate(cues, 1)]
    dest.write_text("\n".join(blocks), encoding="utf-8")
    return len(cues)


def plan(manifest: list[dict]) -> tuple[list[dict], float]:
    """Give every line an absolute start, and every scene a start and an end."""
    t = LEAD_IN
    prev_scene = None
    for m in manifest:
        if prev_scene is not None and m["scene"] != prev_scene:
            t += GAP_SCENE - GAP_LINE
        m["start"] = round(t, 3)
        t += m["seconds"] + GAP_LINE
        prev_scene = m["scene"]
    return manifest, round(t - GAP_LINE + TAIL, 3)


def scenes_of(manifest: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in manifest:
        if not out or out[-1]["name"] != m["scene"]:
            out.append({"name": m["scene"], "start": m["start"], "end": 0.0, "lines": []})
        out[-1]["lines"].append(m)
        out[-1]["end"] = round(m["start"] + m["seconds"], 3)
    return out


def build_audio(manifest: list[dict], total: float, dest: pathlib.Path) -> None:
    """One track, each line laid at its own start. adelay takes milliseconds."""
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for i, m in enumerate(manifest):
        ms = int(m["start"] * 1000)
        inputs += ["-i", str(VOICE / m["file"])]
        filters.append(f"[{i}:a]adelay={ms}|{ms}[a{i}]")
        labels.append(f"[a{i}]")
    chain = (";".join(filters) + ";" + "".join(labels)
             + f"amix=inputs={len(manifest)}:normalize=0:dropout_transition=0[m]"
             + f";[m]apad,atrim=0:{total},loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[out]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", chain,
                    "-map", "[out]", "-c:a", "pcm_s16le", str(dest)], check=True)


def main() -> int:
    mf = VOICE / "manifest.json"
    if not mf.exists():
        print("run scripts/voice.py first", file=sys.stderr)
        return 1
    manifest, total = plan(json.loads(mf.read_text(encoding="utf-8")))
    scenes = scenes_of(manifest)
    print(f"  {len(scenes)} scenes, {total:.1f}s total")
    for s in scenes:
        print(f"    {s['name']:<8} {s['start']:6.2f} -> {s['end']:6.2f}")

    tpl = (MEDIA / "film.template.html").read_text(encoding="utf-8")
    (MEDIA / "film.html").write_text(
        tpl.replace("/*__TIMELINE__*/", json.dumps({"scenes": scenes, "total": total})),
        encoding="utf-8")

    n = write_srt(manifest, MEDIA / "warrant-film.srt")
    print(f"  wrote media/warrant-film.srt  ({n} cues)")

    build_audio(manifest, total, MEDIA / "_film.wav")
    print("  audio track built")

    from playwright.sync_api import sync_playwright
    tmp = MEDIA / "_pw"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--force-color-profile=srgb"])
        ctx = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=2,
                            record_video_dir=str(tmp),
                            record_video_size={"width": W, "height": H},
                            reduced_motion="no-preference")
        pg = ctx.new_page()
        pg.goto((MEDIA / "film.html").resolve().as_uri())
        time.sleep(total + 1.2)
        pg.close()
        ctx.close()
        b.close()
    raw = sorted(tmp.glob("*.webm"))[0]
    shutil.move(str(raw), str(MEDIA / "_film.webm"))
    shutil.rmtree(tmp, ignore_errors=True)
    print("  video recorded")

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(MEDIA / "_film.webm"),
         "-i", str(MEDIA / "_film.wav"),
         "-map", "0:v", "-map", "1:a",
         "-vf", "scale=1920:-2:flags=lanczos,pad=1920:1080:0:(oh-ih)/2:color=0x12161A",
         "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
         str(MEDIA / "warrant-film.mp4")], check=True)
    for scratch in ("_film.webm", "_film.wav"):
        (MEDIA / scratch).unlink(missing_ok=True)
    size = (MEDIA / "warrant-film.mp4").stat().st_size / 1048576
    print(f"  wrote media/warrant-film.mp4  ({size:.1f} MB, {total:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
