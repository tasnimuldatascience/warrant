"""Synthesise the narration, one line per file, and report each line's real duration.

Audio first, visuals second. Timing a video to a *guessed* narration length and then hoping
the voice fits is how a cut ends up with a sentence landing two beats after the thing it
describes. These durations are what the scene timings are built from.

Free and account-free: edge-tts speaks through the same endpoint Edge's read-aloud uses. It
is a build-time dependency for making a video, never a runtime one -- nothing in `warrant`
imports it.

    pip install edge-tts
    python scripts/voice.py            # writes media/_voice/NN.mp3 and prints a timing table
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess

VOICE = "en-US-AndrewNeural"
RATE = "+6%"     # a touch quicker than default; the default reads as an audiobook
OUT = pathlib.Path("media/_voice")

#: One entry per spoken line. `scene` groups lines that share a visual; the builder uses it to
#: hold a scene until its last line has finished rather than cutting on a fixed clock.
LINES: list[tuple[str, str]] = [
    ("hook",    "Ask an A I a legal question, and it will give you an answer. Confidently."),
    ("hook",    "But the law changes."),
    ("change",  "This paragraph said a service agreement had to run at least six months."),
    ("change",  "Until February, when that floor was deleted."),
    ("change",  "So the right answer depends on when you are asking."),
    ("problem", "Most retrieval systems cannot tell you which one you got."),
    ("problem", "This one can."),
    ("what",    "Warrant is a bitemporal retrieval system over federal employment regulation."),
    ("what",    "Every paragraph carries two clocks. When it was the law, "
                "and when the system believed it was."),
    ("split",   "Ask as of twenty twenty-four, and the gate admits nine thousand, "
                "seven hundred and thirty paragraphs."),
    ("split",   "Ask as of today, and it admits ten thousand and seven."),
    ("split",   "Different evidence. Different answer. Both correct."),
    ("gate",    "The date is not a hint in a prompt. It is a filter pushed into the query, "
                "so superseded text is never even a candidate."),
    ("gate",    "Removing that filter costs ninety-six points of wrong-version rate. "
                "Two hundred and twenty wins. Zero losses."),
    ("trace",   "And when it is wrong, every stage recorded what it saw."),
    ("trace",   "So the question is not, why did the model hallucinate. "
                "It is, which stage lost the evidence."),
    ("off",     "Five components in here ship switched off."),
    ("off",     "A reranker. An entailment verifier. A calibrated confidence model. "
                "Multi-hop retrieval. A fine-tuned encoder."),
    ("off",     "Each one built. Each one measured. Each one disabled, "
                "with the p-value that decided it sitting next to the switch."),
    ("close",   "Because measuring something and shipping it anyway is not engineering."),
    ("close",   "Warrant. The code, and every measurement, "
                "including the ones that came out negative."),
]


def duration(path: pathlib.Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


async def main() -> int:
    import edge_tts

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*"):
        stale.unlink()

    manifest = []
    for i, (scene, text) in enumerate(LINES):
        path = OUT / f"{i:02d}.mp3"
        await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(path))
        secs = duration(path)
        manifest.append({"i": i, "scene": scene, "text": text,
                         "file": path.name, "seconds": round(secs, 3)})
        print(f"  {i:02d}  {scene:<8} {secs:5.2f}s  {text[:62]}", flush=True)

    total = sum(m["seconds"] for m in manifest)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  {len(manifest)} lines, {total:.1f}s of speech before gaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
