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

#: One entry per spoken line: (scene, spoken, caption).
#:
#: `scene` groups lines that share a visual; the builder uses it to hold a scene until its last
#: line has finished rather than cutting on a fixed clock.
#:
#: `caption` is the same sentence written for the eye instead of the synthesiser, and is `None`
#: when the two agree. They disagree wherever the voice needs a number spelled out -- a caption
#: reading "nine thousand, seven hundred and thirty" beside a graphic reading 9,730 looks like a
#: transcription error, and "A I" looks like a typo. The audio is unchanged; only the subtitle
#: sidecar reads from this column.
LINES: list[tuple[str, str, str | None]] = [
    ("hook",    "Ask an A I a legal question, and it will give you an answer. Confidently.",
                "Ask an AI a legal question, and it will give you an answer. Confidently."),
    ("hook",    "But the law changes.", None),
    ("change",  "This paragraph said a service agreement had to run at least six months.", None),
    ("change",  "Until February, when that floor was deleted.", None),
    ("change",  "So the right answer depends on when you are asking.", None),
    ("problem", "Most retrieval systems cannot tell you which one you got.", None),
    ("problem", "This one can.", None),
    ("what",    "Warrant is a bitemporal retrieval system over federal employment regulation.",
                None),
    ("what",    "Every paragraph carries two clocks. When it was the law, "
                "and when the system believed it was.", None),
    ("split",   "Ask as of twenty twenty-four, and the gate admits nine thousand, "
                "seven hundred and thirty paragraphs.",
                "Ask as of 2024, and the gate admits 9,730 paragraphs."),
    ("split",   "Ask as of today, and it admits ten thousand and seven.",
                "Ask as of today, and it admits 10,007."),
    ("split",   "Different evidence. Different answer. Both correct.", None),
    ("gate",    "The date is not a hint in a prompt. It is a filter pushed into the query, "
                "so superseded text is never even a candidate.", None),
    ("gate",    "Removing that filter costs ninety-six points of wrong-version rate. "
                "Two hundred and twenty wins. Zero losses.",
                "Removing that filter costs 96 points of wrong-version rate. "
                "220 wins. 0 losses."),
    ("trace",   "And when it is wrong, every stage recorded what it saw.", None),
    ("trace",   "So the question is not, why did the model hallucinate. "
                "It is, which stage lost the evidence.", None),
    ("off",     "Five components in here ship switched off.", None),
    ("off",     "A reranker. An entailment verifier. A calibrated confidence model. "
                "Multi-hop retrieval. A fine-tuned encoder.", None),
    ("off",     "Each one built. Each one measured. Each one disabled, "
                "with the p-value that decided it sitting next to the switch.", None),
    ("close",   "Because measuring something and shipping it anyway is not engineering.", None),
    ("close",   "Warrant. The code, and every measurement, "
                "including the ones that came out negative.", None),
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
    for i, (scene, text, caption) in enumerate(LINES):
        path = OUT / f"{i:02d}.mp3"
        await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(path))
        secs = duration(path)
        manifest.append({"i": i, "scene": scene, "text": text,
                         "caption": caption or text,
                         "file": path.name, "seconds": round(secs, 3)})
        print(f"  {i:02d}  {scene:<8} {secs:5.2f}s  {text[:62]}", flush=True)

    total = sum(m["seconds"] for m in manifest)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  {len(manifest)} lines, {total:.1f}s of speech before gaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
