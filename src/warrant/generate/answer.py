"""Generating a grounded answer from retrieved regulation.

The model does two things it is good at: writing a short answer in plain language, and saying
which retrieved paragraph each sentence came from. It is not asked to do the thing it is bad
at -- producing character offsets -- so the contract is `claim + evidence ids`, and
`warrant.verify.align` turns that into spans afterwards.

Output is constrained by schema, not by hope: the prompt demands a JSON object, the response
is parsed, and a malformed response is retried once at temperature 0 before the answer is
abandoned. Abandoning is a real outcome and it is recorded, because an answer assembled from
a half-parsed response is worse than no answer.

Generation runs on a small local model (Qwen2.5-1.5B-Instruct by default, ~3 GB in fp16).
Everything in this repository has to run on one laptop from a clean clone with no API key,
and a 1.5B model that cites correctly is worth more here than a larger one a reviewer cannot
run. The generator is the last stage, so its ceiling is visible in the failure budget rather
than hidden in a score.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..index.store import Store
from ..retrieve.hybrid import Trace
from ..verify.align import Span, align

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 420
#: Evidence paragraphs offered to the model. More context is not free on a 1.5B model: the
#: cited-chunk accuracy falls off well before the context window does.
MAX_CONTEXT_CHUNKS = 8

SYSTEM = (
    "You answer questions about US federal HR regulation using only the numbered excerpts "
    "provided. You never use outside knowledge and never guess. If the excerpts do not "
    "answer the question, you say so."
)

INSTRUCTIONS = """\
Answer the question using ONLY the excerpts above.

Reply with a single JSON object and nothing else:

{{"claims": [{{"text": "<one sentence>", "evidence": [<excerpt numbers>]}}], \
"answer_found": true}}

Rules:
- Each claim is one plain sentence a person could act on.
- Every claim must list at least one excerpt number it came from.
- Use at most 4 claims.
- If the excerpts do not answer the question, reply {{"claims": [], "answer_found": false}}.

Question: {question}
"""


@dataclass(frozen=True)
class Claim:
    text: str
    evidence: list[str]                       # version ids
    spans: dict[str, Span | None] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        """True when at least one cited chunk yields a locatable supporting span."""
        return any(s is not None for s in self.spans.values())


@dataclass(frozen=True)
class Answer:
    question: str
    as_of: str
    scope: str
    claims: list[Claim]
    answer_found: bool
    cited: dict[str, str]                     # version id -> chunk text
    raw: str = ""
    parse_failed: bool = False

    @property
    def abstained(self) -> bool:
        return not self.answer_found or not self.claims

    @property
    def ungrounded_claims(self) -> list[Claim]:
        return [c for c in self.claims if not c.grounded]

    def text(self) -> str:
        return " ".join(c.text for c in self.claims)


def build_prompt(question: str, excerpts: list[tuple[str, str, str]]) -> list[dict]:
    """``excerpts`` is (version_id, heading, text), presented as numbered blocks.

    Numbered, not addressed by version id: a 1.5B model copies ``630.306#a@2020-08-10``
    wrongly often enough to matter, and a mis-copied citation is indistinguishable from a
    hallucinated one downstream. Small integers map back exactly.
    """
    blocks = []
    for i, (_vid, heading, text) in enumerate(excerpts, start=1):
        head = f" ({heading})" if heading else ""
        blocks.append(f"[{i}]{head} {text}")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(blocks) + "\n\n"
         + INSTRUCTIONS.format(question=question)},
    ]


_JSON = re.compile(r"\{.*\}", re.S)


def parse_response(raw: str,
                   excerpts: list[tuple[str, str, str]]
                   ) -> tuple[list[Claim], bool] | None:
    """Parse the model's JSON into claims with version ids, or None if it is unusable.

    Excerpt numbers outside the offered range are dropped rather than clamped: a citation to
    excerpt 9 when 8 were offered is a hallucinated reference, and silently rewriting it to
    the nearest real one would manufacture grounding.
    """
    match = _JSON.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "claims" not in data:
        return None

    claims: list[Claim] = []
    for entry in data.get("claims") or []:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        ids: list[str] = []
        for n in entry.get("evidence") or []:
            try:
                idx = int(n)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(excerpts):
                ids.append(excerpts[idx - 1][0])
        claims.append(Claim(text=text, evidence=list(dict.fromkeys(ids))))
    return claims, bool(data.get("answer_found", bool(claims)))


def ground(claims: list[Claim], cited: dict[str, str]) -> list[Claim]:
    """Attach a supporting span to every citation, or None where none can be located."""
    return [
        Claim(text=c.text, evidence=c.evidence,
              spans={vid: align(c.text, cited.get(vid, "")) for vid in c.evidence})
        for c in claims
    ]


def excerpts_for(store: Store, trace: Trace, *,
                 limit: int = MAX_CONTEXT_CHUNKS) -> list[tuple[str, str, str]]:
    keys = trace.final[:limit]
    if not keys:
        return []
    rows = {r["version_id"]: r for r in store.db.execute(
        f"SELECT version_id, heading, text FROM chunk WHERE version_id IN "
        f"({','.join('?' * len(keys))})", keys)}
    return [(k, rows[k]["heading"] or "", rows[k]["text"]) for k in keys if k in rows]
