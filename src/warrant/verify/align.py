"""Locating the span of a chunk that supports a claim.

The generator emits **claims plus evidence chunk ids**, never character offsets. Asking a
small model to count characters produces confidently wrong indices: it is a counting task
dressed as a language task, and the failure is silent because a plausible-looking offset
still renders. Selecting which chunk supports a claim is a language task and the model is
good at it; converting that to a character range is arithmetic, and this module does it.

The aligner is allowed to return **no span**. That is not a failure of the aligner, it is a
finding about the answer: the model claimed support from a chunk in which no supporting text
can be located. Silently emitting a whole-chunk span in that case would turn an ungrounded
claim into a grounded-looking one, which is the precise failure the grounding stage exists to
catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Sentence-ish boundaries. Regulatory prose is full of "5 U.S.C. 6304(d)" and "§ 630.306",
#: so a naive split on "." shatters citations; require whitespace and a capital or a
#: paragraph designator to follow.
_BOUNDARY = re.compile(r"(?<=[.;:])\s+(?=[(A-Z])")
_WORD = re.compile(r"[a-z0-9]+")
#: Words too common to count as evidence of support.
_STOP = frozenset("""
a an the and or of to in for on at by is are was were be been being as that this these those
it its with which shall must may not any all such other under upon into from than then when
""".split())

#: Below this share of the claim's content words, the window is not supporting text.
MIN_OVERLAP = 0.30
#: A span may span at most this many consecutive sentences before it stops being a citation.
MAX_WINDOW = 3


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    score: float

    def text_of(self, source: str) -> str:
        return source[self.start:self.end]


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _sentences(text: str) -> list[tuple[int, int]]:
    """Character ranges of sentence-ish units, covering the whole string."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if b > a]


def align(claim: str, source: str, *, min_overlap: float = MIN_OVERLAP,
          max_window: int = MAX_WINDOW) -> Span | None:
    """The smallest window of ``source`` that best supports ``claim``, or None.

    Scored on the share of the *claim's* content words the window covers, not on similarity
    in either direction. Similarity would reward a window that merely looks like the claim;
    the question here is whether the source says what the claim says, so a long window that
    happens to contain every claim word wins over a short one that contains half of them --
    and among windows that cover equally, the shortest wins, because a citation that points
    at three sentences to support six words is not much of a citation.
    """
    wanted = _content_words(claim)
    if not wanted:
        return None
    sentences = _sentences(source)
    if not sentences:
        return None

    best: Span | None = None
    for i in range(len(sentences)):
        for width in range(1, max_window + 1):
            if i + width > len(sentences):
                break
            start, end = sentences[i][0], sentences[i + width - 1][1]
            covered = wanted & _content_words(source[start:end])
            score = len(covered) / len(wanted)
            if score < min_overlap:
                continue
            if best is None or score > best.score + 1e-9 or (
                abs(score - best.score) <= 1e-9 and (end - start) < (best.end - best.start)
            ):
                best = Span(start, end, score)
    return best


def align_all(claim: str, sources: dict[str, str], **kw) -> dict[str, Span | None]:
    """Align a claim against every chunk it cites. A None entry is a grounding failure."""
    return {chunk_id: align(claim, text, **kw) for chunk_id, text in sources.items()}
