"""Serving guardrails: what the API refuses, and what each refusal costs.

**This is not confidentiality.** The corpus is eCFR — published federal law. Nothing here is
secret, nothing can leak, and this module makes no leak-rate claim (ARCHITECTURE.md section 3
says why at length). What it defends is *serving integrity* — that an answer which leaves the
process is addressable, in force, and built from evidence the retriever actually returned —
and *abuse resistance*, which on a public read-only endpoint means cost, not exfiltration.

The threat that was actually measured is resource exhaustion. A query of one token repeated
2,600 times — 15.6 KB, comfortably inside any HTTP parser's limit — took **29,055 ms** of
pinned CPU against 16 ms for a normal query, because FTS5 merges one postings list against
itself once per repeat. Deduplication and a 64-token cap in `warrant.retrieve.hybrid.fts_query`
took it to 3.7 ms.

That fix is real and it is not a guard. It lives inside one ranking stage, so it holds for
exactly as long as every caller goes through `fts_query` — and the same string also reaches
`Retriever._dense`, which embeds it raw. Measured, the encoder turned out to be bounded
already: `bge-small` truncates at 512 tokens, so 2,600 repeats cost 17.9 ms against 11.2 ms
for a real question, linear tokenisation and nothing worse. That is the honest version, and it
is why the door-level bound is justified on a different argument than "the encoder is
exposed": a bound applied at one stage is a bug fix, a bound applied at the door is a
guarantee, and the resource this system is actually short of is not milliseconds of FTS5.

**It is the generation slot.** A degenerate query that gets past the door takes a serialised
19.7 s slot out of a ceiling of three requests per minute. Refusing it costs 23 µs. Every
input rule below is priced that way — not in the FTS5 time it saves, but in the share of the
service's entire capacity it stops one nonsense request from consuming.

Four things are enforced, in the order a request meets them:

    1. input        length, tokens, control characters, homoglyphs, degenerate repetition
    2. prompt       retrieved text is data: chat-template markers neutralised, size bounded
    3. admission    per-client token bucket, and a per-request cost bound
    4. output       every claim cites evidence; every citation was retrieved and is in force

The SLO being defended is the one the whole service is pinned to: **generation runs at 21.3
tok/s unbatched, about three requests per minute**. Retrieval is not the constraint — 18.4 ms
p50. So the limiter's job is not fairness (a semaphore in `api` already bounds concurrency);
it is to make over-ceiling load cost a dict lookup instead of a 20-second queued thread.

Output validation **fails loudly**. A response that cites a chunk the retriever never returned
is withheld with a 500, not repaired and served: repairing it would mean deciding which of the
model's citations to believe, and the entire grounding contract exists to avoid that decision.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Send

from ..generate.answer import MAX_CONTEXT_CHUNKS, MAX_NEW_TOKENS, Answer
from ..index.store import Store
from ..index.store import now as utc_now
from ..retrieve.hybrid import MAX_QUERY_TOKENS, fts_query

# -- 1. input ---------------------------------------------------------------------------

#: Characters, measured on the string as it arrived rather than on what survives
#: normalisation: NFKC over 15.6 KB is itself the work an attacker is buying. Measured on the
#: 56 hand-written benchmark questions the longest is 82 characters, so this is 6x headroom.
MAX_QUERY_CHARS = 512
#: Below this there is no question, only a prefix. `api.ask` already declares min_length=2.
MIN_QUERY_CHARS = 2

#: Repetition is only degenerate once there is enough of it to be deliberate. The longest
#: benchmark question is 15 tokens, so nothing a person writes reaches this floor. Inside the
#: character cap the cost of serving one of these is modest -- 85 repeats scan in 44.4 ms
#: against 21.1 ms for a real question -- so this rule is not paying for FTS5 time. It is
#: paying for the 19.7 s generation slot the request would go on to occupy.
REPETITION_TOKEN_FLOOR = 24
#: distinct/total below this is not a question. The *tightest* benchmark question sits at
#: 0.833 — "How long can a term appointment last?" — so the margin is 3.3x, and 0 of 56
#: real questions trip it.
MIN_DISTINCT_RATIO = 0.25

#: Control and format characters are replaced by a space, never deleted. Deleting them joins
#: the halves of a split word ("an​nual" -> "annual"), which is exactly the rewrite an
#: attacker wants; replacing splits it, which is what the character honestly is.
_KEEP_CONTROL = frozenset("\t\n\r")

#: Homoglyphs NFKC does *not* fold, because they are different scripts rather than different
#: forms of one character. The reason to fold them here is correctness, not spoofing: measured
#: against the corpus, Cyrillic "аnnual" and fullwidth "ａnnual" each match **0** chunks where
#: "annual" matches 100, so retrieval returns nothing and the answer renders as a confident
#: "nothing is in force" — the same failure mode `api._date` was written to close for
#: as_of=2021-13-45. Applied to the *query*
#: only. 2,886 of 13,145 chunks contain non-ASCII (section signs, em dashes, ligatures) and
#: rewriting stored regulation text would corrupt citations.
_CONFUSABLES = str.maketrans({
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J",
    # Greek
    "α": "a", "ν": "v", "ο": "o", "ρ": "p", "υ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})

#: What `fts_query` is allowed to have produced: quoted alphanumeric runs joined by OR, or the
#: empty-match sentinel. Checked on every request rather than trusted, because the escape is
#: one string operation away from letting FTS5's own query language reach SQLite. Measured
#: against the corpus, unescaped: 26 prefix wildcards 166.4 ms, a 59-term NEAR chain 36.2 ms,
#: and `annual "leave` an OperationalError — a 500 handed to the caller for a quotation mark.
_LITERAL_FTS = re.compile(r'^(?:""|"[0-9A-Za-z]+"(?: OR "[0-9A-Za-z]+")*)$')


class GuardError(Exception):
    """Base for everything this module refuses. Never raised directly."""


class Rejected(GuardError):
    """The request was refused before it cost anything. The caller's problem, so 4xx.

    ``reason`` is a stable machine token and ``detail`` is the sentence a caller reads. They
    are separate because a client that has to regex an English sentence to tell "too long"
    from "too many repeats" will keep working right up until the sentence is reworded.
    """

    def __init__(self, status: int, reason: str, detail: str, *, retry_after: int | None = None):
        super().__init__(detail)
        self.status = status
        self.reason = reason
        self.detail = detail
        self.retry_after = retry_after

    def as_http(self) -> HTTPException:
        headers = {"Retry-After": str(self.retry_after)} if self.retry_after else None
        return HTTPException(self.status, f"{self.reason}: {self.detail}", headers=headers)


class ResponseWithheld(GuardError):
    """An assembled answer failed output validation and was not served.

    5xx, not 4xx. The caller asked a well-formed question; the pipeline produced something the
    system cannot stand behind, which is the server's failure to admit rather than the
    caller's to fix.
    """

    def __init__(self, problems: Sequence[Problem]):
        super().__init__("; ".join(f"{p.kind}: {p.detail}" for p in problems))
        self.problems = list(problems)


@dataclass(frozen=True, slots=True)
class Question:
    """A query that has passed the door, in the exact form the pipeline should see.

    ``text`` is what retrieval and the encoder get, and what the response should echo: an
    answer to the folded query is not an answer to the bytes that arrived, and reporting the
    raw form would attribute the answer to a question nobody answered.
    """

    raw: str
    text: str
    tokens: list[str]
    fts: str
    #: Normalisation changed the string — folded homoglyphs, stripped controls, collapsed
    #: whitespace. Worth surfacing: it is the difference between "no results" and "no results
    #: for the question you think you asked".
    normalised: bool = False
    #: Tokens past `MAX_QUERY_TOKENS` were dropped. The same cap `fts_query` applies, applied
    #: where it can be reported instead of happening silently inside a ranking stage.
    truncated: bool = False


def normalise(text: str) -> str:
    """NFKC, control and format characters to spaces, cross-script homoglyphs to ASCII.

    NFKC alone is not enough and neither is the confusable table alone. NFKC folds width and
    ligatures (``ｄｕｔｙ``, ``ﬁ``) and leaves Cyrillic ``а`` untouched; the table folds
    ``а`` and would have to enumerate every fullwidth codepoint to do the first job.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch if ch in _KEEP_CONTROL or unicodedata.category(ch) not in ("Cc", "Cf") else " "
        for ch in text
    )
    return " ".join(text.translate(_CONFUSABLES).split())


def tokenise(text: str) -> list[str]:
    """The same split `fts_query` performs, before dedup and the cap.

    Deliberately duplicated from `fts_query` in the one respect that matters: this keeps the
    repeats, because the repeats are the measurement. `fts_query` cannot report degenerate
    repetition — by the time it returns, it has already thrown the evidence away.
    """
    return ("".join(c if c.isalnum() else " " for c in text)).split()


def check_question(raw: object, *, max_chars: int = MAX_QUERY_CHARS,
                   max_tokens: int = MAX_QUERY_TOKENS) -> Question:
    """Validate and normalise one question, or refuse it. Never raises anything else.

    Order is cost order: the length test reads ``len``, the repetition test needs the tokens,
    and the FTS5 self-check needs the escape. A 15.6 KB query is refused after one comparison.
    """
    if not isinstance(raw, str):
        raise Rejected(422, "not_a_string", f"expected a string question, got {type(raw).__name__}")
    if len(raw) > max_chars:
        raise Rejected(422, "too_long",
                       f"question is {len(raw)} characters; the limit is {max_chars}")

    text = normalise(raw)
    if len(text) < MIN_QUERY_CHARS:
        raise Rejected(422, "too_short",
                       f"question is {len(text)} characters after normalisation; "
                       f"the minimum is {MIN_QUERY_CHARS}")

    tokens = tokenise(text)
    if not tokens:
        # Refused rather than answered. `fts_query` turns a punctuation-only query into '""',
        # which matches no row, and the endpoint would return 200 with an empty evidence list
        # that the UI renders as "nothing is in force on this date".
        raise Rejected(422, "no_terms", "question contains no searchable term")

    distinct = len(set(tokens))
    if len(tokens) >= REPETITION_TOKEN_FLOOR and distinct / len(tokens) < MIN_DISTINCT_RATIO:
        raise Rejected(422, "degenerate_repetition",
                       f"{len(tokens)} tokens but only {distinct} distinct; "
                       f"the minimum ratio is {MIN_DISTINCT_RATIO}")

    unique = list(dict.fromkeys(tokens))
    fts = fts_query(text, max_tokens=max_tokens)
    if not _LITERAL_FTS.match(fts):
        # Unreachable while `fts_query` quotes every token, which is the point: this asserts
        # the escape on the request rather than on the day it was written.
        raise Rejected(422, "unescapable", "question could not be reduced to a literal query")

    return Question(raw=raw, text=text, tokens=unique[:max_tokens], fts=fts,
                    normalised=text != raw, truncated=len(unique) > max_tokens)


# -- 2. prompt assembly -----------------------------------------------------------------

#: Chat-template control tokens, across the families a local generator might use. The
#: generator inserts retrieved text into a chat template, so a chunk containing ``<|im_end|>``
#: would not be text *inside* the user turn — it would end the turn, and everything after it
#: would be read as a new message with a role of its own choosing. That is the one genuinely
#: structural prompt-injection hole in this pipeline.
#:
#: 0 of 13,145 in-force chunks contain any of these, so neutralisation is lossless on eCFR.
#: It is here for `sources/html.py`: OPM guidance pages are fetched from the web, and a page
#: is a document an author outside this project controls.
_ROLE_MARKERS = re.compile(r"<\|[A-Za-z0-9_]{1,32}\|>|</?s>|\[/?INST\]|<</?SYS>>", re.I)

#: Per-excerpt character cap. The longest in-force chunk is 7,948 characters, so this
#: truncates nothing in the corpus today and bounds the HTML path, where "one chunk" is
#: whatever a fetched page turned out to be.
MAX_EXCERPT_CHARS = 8_000
#: Whole-prompt cap. Measured over every contiguous 16-chunk window of the corpus, an
#: assembled prompt is p50 3,586 / p99 9,306 / max 12,172 characters, so this is 2x the
#: largest prompt eCFR can produce and truncates nothing. The worst case without a cap is
#: 16 x 7,948 = 127,168, and attention is quadratic in what it is given.
MAX_PROMPT_CHARS = 24_000
#: Generation throughput, measured unbatched. The number this entire service is pinned to.
GENERATION_TOK_PER_S = 21.3


@dataclass(frozen=True, slots=True)
class Cost:
    """What one request is allowed to cost, in the only unit that binds: generation seconds.

    Prefill is not modelled because it was not measured; the prompt is capped outright
    instead. Guessing a prefill rate and reporting the product as a prediction would be a
    number with the shape of a measurement and none of the content.
    """

    excerpts: int
    prompt_chars: int
    max_new_tokens: int = MAX_NEW_TOKENS

    @property
    def decode_s(self) -> float:
        """Seconds of decode at the measured ceiling. 420 tokens at 21.3 tok/s = 19.7 s."""
        return self.max_new_tokens / GENERATION_TOK_PER_S

    def check(self, deadline_s: float) -> None:
        if self.decode_s > deadline_s:
            raise Rejected(503, "over_budget",
                           f"decode alone needs {self.decode_s:.1f}s of a {deadline_s:.0f}s "
                           f"budget", retry_after=RETRY_AFTER_S)


@dataclass(frozen=True, slots=True)
class Prompt:
    """Bounded, neutralised excerpts, plus what had to be done to get them that way."""

    excerpts: list[tuple[str, str, str]]
    dropped: int = 0
    truncated: int = 0
    neutralised: int = 0

    @property
    def chars(self) -> int:
        return sum(len(t) for _, _, t in self.excerpts)

    def cost(self, max_new_tokens: int = MAX_NEW_TOKENS) -> Cost:
        return Cost(len(self.excerpts), self.chars, max_new_tokens)


def neutralise(text: str) -> tuple[str, bool]:
    """Strip chat-template control tokens from retrieved text. Returns (text, changed).

    Retrieved text is data. This does not try to detect instructions in it — "ignore previous
    instructions" is a sentence, and a filter that removes sentences from federal regulation
    on suspicion is a correctness bug wearing a security badge. It removes only the tokens
    that would stop the text *being* data, by closing the turn it is quoted inside.
    """
    cleaned, n = _ROLE_MARKERS.subn(" ", text)
    return (cleaned, True) if n else (text, False)


def bound_excerpts(excerpts: Iterable[tuple[str, str, str]], *,
                   max_excerpts: int = MAX_CONTEXT_CHUNKS,
                   max_excerpt_chars: int = MAX_EXCERPT_CHARS,
                   max_prompt_chars: int = MAX_PROMPT_CHARS) -> Prompt:
    """Cap the context window's input, and neutralise each excerpt on the way through.

    Truncation is per excerpt *before* the whole-prompt cap, so a single outsized chunk cannot
    starve the fifteen behind it. Excerpts that do not fit are dropped from the tail, which is
    fusion's least-preferred end; dropping from the head would discard the ranking.
    """
    kept: list[tuple[str, str, str]] = []
    dropped = truncated = neutralised = 0
    total = 0
    for i, (vid, heading, text) in enumerate(excerpts):
        if i >= max_excerpts:
            dropped += 1
            continue
        text, changed = neutralise(text)
        neutralised += changed
        if len(text) > max_excerpt_chars:
            text = text[:max_excerpt_chars]
            truncated += 1
        if total + len(text) > max_prompt_chars:
            dropped += 1
            continue
        total += len(text)
        kept.append((vid, heading, text))
    return Prompt(kept, dropped=dropped, truncated=truncated, neutralised=neutralised)


# -- 3. admission -----------------------------------------------------------------------

#: One client's sustained answer rate, set to the server's own measured ceiling: 21.3 tok/s
#: unbatched is 0.051 req/s is 3.06 requests/minute. A client inside the ceiling is never
#: limited; a client above it is refused for the cost of a dict lookup rather than admitted
#: into a 20-second queue wait it will lose anyway.
ANSWER_RATE_PER_S = 3.0 / 60.0
ANSWER_BURST = 3
#: Reads are a different order: 18.4 ms p50, and aggregate retrieval throughput peaks at
#: 66 QPS across 4 threads. 10/s sustained caps one client at ~15% of that ceiling, which
#: leaves the timeline and diff views — which fire several requests per interaction — unlimited
#: in practice.
READ_RATE_PER_S = 10.0
READ_BURST = 20
#: Matches `api.RETRY_AFTER_S`. Duplicated rather than imported: importing `api` from here
#: would be a cycle, since `api` is what wires this module in.
RETRY_AFTER_S = 30
#: Distinct client keys held. Unbounded is the failure this project has already had once — an
#: unbounded dict keyed on caller-controlled data, ~1.8 KB an entry, held for the life of the
#: process (`hybrid.Retriever._dense`). A bucket is ~120 bytes, so this is ~0.5 MB, and the
#: LRU victim is by construction the client that has gone quietest.
MAX_TRACKED_CLIENTS = 4096


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """A token bucket per client, with the clock injected and the key set bounded.

    ``clock`` is a parameter because a rate limiter tested against ``time.monotonic`` is
    tested by sleeping, and a suite that sleeps is a suite that is flaky on a loaded machine
    and slow on an idle one. Every test here advances a counter instead.

    Thread-safe: `api` dispatches synchronous endpoints into anyio's threadpool (4 tokens), so
    two requests genuinely read and write these buckets at once.
    """

    def __init__(self, rate_per_s: float, burst: int, *,
                 capacity: int = MAX_TRACKED_CLIENTS,
                 clock: Callable[[], float] = time.monotonic):
        if rate_per_s <= 0 or burst <= 0:
            raise ValueError("rate_per_s and burst must both be positive")
        self.rate_per_s = rate_per_s
        self.burst = burst
        self.capacity = capacity
        self._clock = clock
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()
        self.allowed = 0
        self.refused = 0
        self.evicted = 0

    def wait(self, key: str) -> float:
        """0.0 if this request may proceed, else seconds until it could.

        Consumes a token on success and nothing on failure: a refused request must not push
        its own next attempt further out, or a client hammering a closed door can never
        reopen it.
        """
        with self._lock:
            now = self._clock()
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.burst), updated=now)
                self._buckets[key] = bucket
                self._evict()
            else:
                self._buckets.move_to_end(key)
                elapsed = max(0.0, now - bucket.updated)
                bucket.tokens = min(float(self.burst),
                                    bucket.tokens + elapsed * self.rate_per_s)
                bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                self.allowed += 1
                return 0.0
            self.refused += 1
            return (1.0 - bucket.tokens) / self.rate_per_s

    def allow(self, key: str) -> bool:
        return self.wait(key) <= 0.0

    def _evict(self) -> None:
        while len(self._buckets) > self.capacity:
            self._buckets.popitem(last=False)
            self.evicted += 1

    def stats(self) -> dict[str, int]:
        """Kept apart on purpose: "refused" and "evicted" have different fixes."""
        return {"clients": len(self._buckets), "allowed": self.allowed,
                "refused": self.refused, "evicted": self.evicted}


def client_key(scope: dict[str, Any], *, trust_forwarded: bool = False) -> str:
    """Identify the caller. The peer address, unless a trusted proxy is declared.

    ``X-Forwarded-For`` is off by default because it is set by the client. Trusting it without
    a proxy in front turns the limiter into a header-shaped opt-out that still reports
    success, which is worse than having no limiter: the counters say the policy is working.
    Turn it on only where something upstream overwrites the header.
    """
    if trust_forwarded:
        for name, value in scope.get("headers", ()):
            if name == b"x-forwarded-for":
                first = value.decode("latin-1").split(",")[0].strip()
                if first:
                    return first
    client = scope.get("client")
    return client[0] if client else "unknown"


class RateLimitMiddleware:
    """Token-bucket admission in front of the endpoints that cost more than a dict lookup.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so a refusal does not allocate a task group
    and a streaming response body per rejected request — the whole point is that saying no is
    cheap.

    ``/health`` and ``/ready`` are never limited. A liveness probe that gets a 429 restarts a
    process that was working, and a readiness probe that gets one removes a healthy instance
    from rotation at exactly the moment the remaining ones are busiest.
    """

    def __init__(self, app: ASGIApp, *,
                 answer: RateLimiter | None = None, read: RateLimiter | None = None,
                 answer_paths: Sequence[str] = ("/api/ask",),
                 prefix: str = "/api/", exempt: Sequence[str] = ("/health", "/ready"),
                 trust_forwarded: bool = False, enabled: bool = True):
        self.app = app
        self.answer = answer or RateLimiter(ANSWER_RATE_PER_S, ANSWER_BURST)
        self.read = read or RateLimiter(READ_RATE_PER_S, READ_BURST)
        self.answer_paths = frozenset(answer_paths)
        self.prefix = prefix
        self.exempt = frozenset(exempt)
        self.trust_forwarded = trust_forwarded
        self.enabled = enabled

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if not self.enabled or scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if path in self.exempt or not path.startswith(self.prefix):
            return await self.app(scope, receive, send)
        limiter = self.answer if path in self.answer_paths else self.read
        wait = limiter.wait(client_key(scope, trust_forwarded=self.trust_forwarded))
        if wait <= 0.0:
            return await self.app(scope, receive, send)
        await _refuse(send, wait, limiter.rate_per_s)


async def _refuse(send: Send, wait: float, rate_per_s: float) -> None:
    """429 with a Retry-After the client can actually act on.

    The header is the bucket's own arithmetic, rounded up, rather than a fixed constant: a
    client told to come back in 30 s when a token arrives in 2 s wastes 28 s of a ceiling that
    is already only three requests a minute.
    """
    retry = max(1, math.ceil(wait))
    body = json.dumps({
        "detail": f"rate_limited: {rate_per_s * 60:.0f} requests/minute sustained; "
                  f"retry in {retry}s"}).encode()
    await send({"type": "http.response.start", "status": 429,
                "headers": [(b"content-type", b"application/json"),
                            (b"retry-after", str(retry).encode()),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


# -- 4. output --------------------------------------------------------------------------

#: A version id: ``630.306#a@2020-08-10``. Checked for shape before it is looked up so that a
#: malformed id is reported as malformed rather than as "no such chunk", which is the same
#: message a retracted one gets.
_VERSION_ID = re.compile(r"^[^@\s]+@\d{4}-\d{2}-\d{2}$")

#: Character offsets smuggled into claim prose. The contract is claim + evidence id, and spans
#: are computed afterwards by `verify.align` precisely because a small model counting
#: characters produces confidently wrong indices (ARCHITECTURE.md section 5). An offset in the
#: answer text is a number no stage in this pipeline computed.
_BARE_OFFSET = re.compile(
    r"\[\s*\d+\s*[:,–-]\s*\d+\s*\]"
    r"|\b(?:char|chars|character|characters|offset|offsets|position|positions)\b"
    r"\s*:?\s*\d+\s*(?:[:,–-]|to)\s*\d+",
    re.I)


@dataclass(frozen=True, slots=True)
class Problem:
    kind: str
    detail: str


def in_force_versions(store: Store, version_ids: Sequence[str], *, as_of: str,
                      system_time: str | None = None) -> set[str]:
    """Which of these versions the store believes and holds in force on ``as_of``.

    Both predicates, together, because they fail differently: ``system_to`` catches a citation
    to text that has since been retracted or reparsed, and ``valid_to`` catches a citation to
    a version that was never in force on the date asked. An answer built at ``final_k`` from a
    correctly predicated retrieval satisfies both; one that does not, did not come from that
    retrieval.
    """
    ids = [v for v in dict.fromkeys(version_ids) if _VERSION_ID.match(v)]
    if not ids:
        return set()
    sys_t = system_time or utc_now()
    marks = ",".join("?" * len(ids))
    rows = store.db.execute(
        f"SELECT version_id FROM chunk WHERE version_id IN ({marks}) "
        "AND system_from <= ? AND (system_to IS NULL OR system_to > ?) "
        "AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)",
        [*ids, sys_t, sys_t, as_of, as_of])
    return {r["version_id"] for r in rows}


def validate_answer(answer: Answer, *, retrieved: Sequence[str],
                    in_force: Iterable[str] | None = None) -> list[Problem]:
    """Everything wrong with this answer, or an empty list. Never raises.

    ``retrieved`` is the set the generator was actually offered — `Trace.final`, sliced the
    same way `excerpts_for` slices it. ``in_force`` is the subset the store still stands
    behind at the requested date; pass None to check addressability only, which is what a
    caller without a store (a replay, a test) can honestly check.

    An abstention is valid by construction: no claims, nothing to cite, nothing to check.
    Failing it would make declining to answer the riskiest thing the system can do.
    """
    offered = set(retrieved)
    standing = None if in_force is None else set(in_force)
    problems: list[Problem] = []

    if answer.answer_found and not answer.claims:
        problems.append(Problem(
            "inconsistent_abstention", "answer_found is true but no claim was produced"))

    for i, claim in enumerate(answer.claims, start=1):
        where = f"claim {i}"
        if not claim.text.strip():
            problems.append(Problem("empty_claim", f"{where} has no text"))
        if not claim.evidence:
            # ARCHITECTURE.md section 9: every claim in an emitted answer carries at least one
            # evidence id. Stated there as an invariant; enforced here on the serving path.
            problems.append(Problem("no_evidence", f"{where} cites nothing"))
        if _BARE_OFFSET.search(claim.text):
            problems.append(Problem(
                "offset_citation", f"{where} carries a character offset; this system cites "
                                   "by evidence id and computes spans afterwards"))
        for vid in claim.evidence:
            if not _VERSION_ID.match(vid):
                problems.append(Problem("malformed_evidence", f"{where} cites {vid!r}"))
            elif vid not in offered:
                # The one that matters for injection: the model can only address a chunk by an
                # excerpt number the prompt offered, so an id outside `retrieved` means the
                # mapping was wrong, not that the model was clever.
                problems.append(Problem(
                    "unretrieved_evidence", f"{where} cites {vid}, which was not retrieved"))
            elif standing is not None and vid not in standing:
                problems.append(Problem(
                    "not_in_force", f"{where} cites {vid}, which is not in force as asked"))
    return problems


def check_answer(answer: Answer, *, retrieved: Sequence[str],
                 in_force: Iterable[str] | None = None) -> Answer:
    """`validate_answer`, but the failure is loud. Returns the answer it approved.

    Nothing is dropped, repaired or downgraded. Removing the offending claim and serving the
    rest would mean deciding which of the model's citations to believe on the evidence of the
    model's citations, and the response would carry no sign that it had happened.
    """
    problems = validate_answer(answer, retrieved=retrieved, in_force=in_force)
    if problems:
        raise ResponseWithheld(problems)
    return answer


def check_answer_against(store: Store, answer: Answer, *, retrieved: Sequence[str],
                         as_of: str, system_time: str | None = None) -> Answer:
    """`check_answer` with the in-force set resolved from the store. One indexed query."""
    cited = [v for claim in answer.claims for v in claim.evidence]
    in_force = in_force_versions(store, cited, as_of=as_of, system_time=system_time)
    return check_answer(answer, retrieved=retrieved, in_force=in_force)


# -- FastAPI shapes ---------------------------------------------------------------------


def question_param(
    q: str = Query(min_length=MIN_QUERY_CHARS, max_length=MAX_QUERY_CHARS,
                   description="The question. Normalised (NFKC, homoglyphs folded) before "
                               "retrieval; capped at 64 distinct terms."),
) -> Question:
    """Dependency form of `check_question`. ``Depends(question_param)``.

    The bounds are declared on ``Query`` as well as enforced in `check_question` so that they
    appear in the OpenAPI document. Declaring them only in the schema would be documentation;
    enforcing them only in code would be an undocumented 422.
    """
    try:
        return check_question(q)
    except Rejected as exc:
        raise exc.as_http() from exc


#: For handlers that want the dependency without spelling out ``Depends``.
QuestionParam = Depends(question_param)


async def guard_error_handler(_request: Any, exc: Exception) -> JSONResponse:
    """Map guard errors onto the statuses `api` already uses. Register on `GuardError`.

    A withheld response is a 500 with the reasons attached rather than a generic error: the
    caller cannot fix it, but the operator reading the log needs to know it was withheld
    deliberately and which check refused it.
    """
    if isinstance(exc, Rejected):
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse({"detail": f"{exc.reason}: {exc.detail}"},
                            status_code=exc.status, headers=headers)
    if isinstance(exc, ResponseWithheld):
        return JSONResponse(
            {"detail": "response withheld by output validation",
             "problems": [{"kind": p.kind, "detail": p.detail} for p in exc.problems]},
            status_code=500)
    raise exc


@dataclass
class Guards:
    """The two limiters an app owns, so a test or an endpoint can read their counters."""

    answer: RateLimiter = field(
        default_factory=lambda: RateLimiter(ANSWER_RATE_PER_S, ANSWER_BURST))
    read: RateLimiter = field(
        default_factory=lambda: RateLimiter(READ_RATE_PER_S, READ_BURST))
    #: Off only where a suite asserts something else. Starlette's ``TestClient`` reports every
    #: caller as the same client, so one module's thirty tests are one client at thirty times
    #: the ceiling, and `tests/test_api.py` would start failing on admission control instead
    #: of on the thing each test is about. The limiter has its own suite; nothing is lost.
    #: Never a production setting -- the default is on, and it takes an argument to turn off.
    enabled: bool = True

    def stats(self) -> dict[str, dict[str, int]]:
        return {"answer": self.answer.stats(), "read": self.read.stats()}
