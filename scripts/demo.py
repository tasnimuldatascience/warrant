"""Drive the demo against a running server, so a recording does not depend on typing.

Fumbling a live demo is the usual reason a good system films badly. This runs the exact
sequence in `docs/DEMO.md` against the real API and prints it at a readable pace, so the
terminal segments can be recorded in one take.

    make serve            # in one terminal, then ask one question so the model is warm
    python scripts/demo.py

Nothing here is a mock. Every number printed came back from the server on the run you are
watching, which is the point: a demo that cannot be re-run in front of someone is a video, not
a demonstration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
PACE = 0.9

# A Windows console defaults to cp1252, which cannot encode the box rule below -- the demo
# would die on its own first heading. Reconfiguring beats downgrading to ASCII: this is meant
# to be filmed.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):  # pragma: no cover - a stream that cannot be reconfigured
        pass


def say(line: str = "", pace: float = PACE) -> None:
    print(line, flush=True)
    time.sleep(pace)


def rule(title: str) -> None:
    say()
    say(f"\033[1m{title}\033[0m")
    say("─" * min(78, max(20, len(title) + 4)), pace=0.3)


def get(path: str, **params: str) -> dict:
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}" if params else f"{BASE}{path}"
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 - localhost only
        return json.loads(r.read())


def stream(path: str, **params: str):
    """Yield (event, data) as the server sends them, so the pacing on screen is the real one."""
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    event = None
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 - localhost only
        for raw in r:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                yield event, json.loads(line[6:])


def main() -> int:
    global BASE, PACE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--pace", type=float, default=PACE,
                    help="seconds between lines; 0 for a fast sanity check")
    args = ap.parse_args()

    BASE, PACE = args.base, args.pace

    try:
        meta = get("/api/meta")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"no server at {BASE} -- run `make serve` first ({exc})", file=sys.stderr)
        return 1

    rule("The corpus")
    say(f"  {meta['chunks']:,} chunk versions   {meta['part_count']} CFR parts   "
        f"{meta['earliest']} to {meta['latest']}")

    # -- 1. the same question, two dates ---------------------------------------------
    q = "How long may a recruitment incentive service agreement run?"
    rule("One question, two dates")
    say(f"  {q}")
    for date in ("2024-06-01", "2026-08-26"):
        r = get("/api/ask", q=q, as_of=date, generate="false")
        top = r["evidence"][0] if r["evidence"] else None
        say(f"\n  as of {date}")
        if top:
            say(f"    {top['chunk_id']}  in force {top['valid_from']} -> "
                f"{top['valid_to'] or 'now'}")
            say(f"    {top['text'][:150]}...")

    # -- 2. the order things arrive in -----------------------------------------------
    rule("Evidence first, prose second")
    q2 = "By when must restored annual leave be scheduled?"
    say(f"  {q2}   as of 2021-06-01")
    started = time.perf_counter()
    trace_id = None
    for event, data in stream("/api/ask/stream", q=q2, as_of="2021-06-01"):
        at = (time.perf_counter() - started) * 1000
        if event == "retrieval":
            # Two clocks, and the difference matters. `at` is what a client sees, connection
            # setup and JSON included; `timings.total` is what retrieval itself cost. Quoting
            # the second as if it were the first is the kind of number this repository has
            # already had to correct once.
            server = data["timings"].get("total", 0.0)
            say(f"    {at:7.0f}ms  retrieval   {data['admitted']:,} paragraphs admitted "
                f"({server:.0f}ms in the retriever, the rest is HTTP)", pace=0.2)
        elif event == "evidence":
            say(f"    {at:7.0f}ms  evidence    {len(data)} chunks, cited and readable",
                pace=0.2)
        elif event == "status":
            say(f"    {at:7.0f}ms  generating  the model starts here", pace=0.2)
        elif event == "claim":
            say(f"    {at:7.0f}ms  claim       {data['text'][:64]}...", pace=0.2)
        elif event == "done":
            trace_id = data.get("trace_id")
            say(f"    {at:7.0f}ms  done        trace {trace_id}", pace=0.2)
        elif event == "error":
            say(f"    error: {data}")
            return 1

    # -- 3. the follow-up whose date cannot move -------------------------------------
    if trace_id:
        rule("A follow-up, pinned to that same date")
        say("  what is the exception in paragraph (b)?")
        for event, data in stream("/api/ask/followup/stream", trace_id=trace_id,
                                  q="what is the exception in paragraph (b)?"):
            if event == "pinned":
                say(f"    pinned to {data['as_of']}  ·  {data['evidence_count']} chunks  ·  "
                    f"{data['scope']}", pace=0.4)
                say("    (this endpoint has no date parameter: nothing can send a different "
                    "one)", pace=0.4)
            elif event == "claim":
                say(f"    claim   {data['text'][:70]}...", pace=0.3)
            elif event == "done":
                say(f"    done    {data.get('kind')}", pace=0.3)
            elif event == "error":
                say(f"    error: {data}")

        rule("And it survives a reload")
        again = get(f"/api/exchange/{trace_id}")
        say(f"  {BASE}/#/ask/{trace_id}")
        say(f"  reopens: as of {again['as_of']}, {len(again['evidence'])} chunks, "
            "the same evidence it was answered from")

    rule("Where the answer came from")
    say("  Every stage recorded what it saw and what it cost, so a wrong answer has an")
    say("  address rather than an excuse.  ->  the Trace screen, or `warrant replay show`")
    say()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
