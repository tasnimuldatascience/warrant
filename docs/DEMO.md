# Demo script — three minutes

For a YouTube upload and a LinkedIn post. Written to be spoken, not read: short sentences,
no adjectives doing work a number could do, and nothing claimed that the repository does not
measure.

**Before you record:** `make serve`, then open `http://127.0.0.1:8000/` and ask one throwaway
question so the model is warm. The first generation loads 2.9 GB and takes about thirty
seconds; every one after is ~6.6 s. Recording that first load makes the system look four times
slower than it is.

Record at 1440×900 or wider. The interface sets a 106-column floor and the tables elide below
it.

---

## 0:00 – 0:14 · Cold open, no narration

Full screen on the hero. Drag the date control from 2024 to 2026 and let the answer change on
its own. Two seconds of silence after it lands.

> *No voice. Let it do the thing.*

---

## 0:14 – 0:40 · What it is

**On screen:** the hero, still. Then scroll to show the question box.

> This answers questions about US federal employment regulation.
>
> That sounds narrow. Here is why it isn't. The law changes. The paragraph you just watched
> change was amended in August 2026, and both answers are correct — one of them in 2024 and
> the other one now.
>
> Almost every retrieval system I have seen would give you one of those and not tell you which.

---

## 0:40 – 1:15 · Ask something, and watch the order

**On screen:** type a real question. `By when must restored annual leave be scheduled?`
Set the date to 2021-06-01. Submit. **Do not cut** — let the evidence land and the answer
lag behind it.

> Watch what arrives first.
>
> The citations are on screen in about half a second. The written answer takes six and a half.
>
> Retrieval itself is eighteen milliseconds of that — the rest is the browser and the network.
> I say both numbers because the small one is the one everybody quotes and the large one is
> the one you actually experience.
>
> Most systems put a spinner over both and show you nothing for six seconds. This one gives
> you the law immediately and the prose when it is ready — because if you are a federal HR
> specialist, the citation *is* the answer. The paragraph is what you act on.

---

## 1:15 – 1:45 · The follow-up, and the date that cannot move

**On screen:** scroll to section 04. Ask *"what's the exception in paragraph (b)?"* Point at
the pinned date on the turn.

> Now a follow-up. Notice the date on this turn — it is the same date as the question above,
> and it says so.
>
> This is the part I would push back on if someone asked me for a chatbot here. If you ask
> about 2019 and then say "and what about the exception", a normal chat interface has to guess
> which year you meant. Guess wrong and you have quoted repealed law with total confidence.
>
> So the follow-up endpoint has no date parameter at all. Not "we remember to pass it
> through" — there is no way to send a different one.

---

## 1:45 – 2:10 · The timeline

**On screen:** Timeline, section 630.306. Scrub across the amendment boundaries.

> Every version of every section, and which one was law on any given day.
>
> The stamped ones are repealed. About half of what is in this corpus is text that no longer
> applies, and if a system cannot show you that at a glance, it will eventually quote it to
> you as if it were current.

---

## 2:10 – 2:45 · The part I actually care about

**On screen:** Trace. The stage ladder.

> This is the screen the whole project is built around.
>
> Nine thousand six hundred paragraphs were eligible. A hundred survived search. Sixteen
> reached the model. Every stage recorded what it saw and what it cost.
>
> So when the answer is wrong — and it will sometimes be wrong — the question is not "why is
> the AI hallucinating". The question is which stage lost the evidence. That is answerable,
> and it is the difference between a demo and something you could operate.

---

## 2:45 – 3:05 · The uncomfortable slide

**On screen:** the README, on the baselines table or the "deliberately does not do" section.

> Last thing, and it is the part I would want to be asked about.
>
> Five components in here are switched **off**. The reranker, an entailment verifier, a
> calibrated confidence model, multi-hop retrieval, and a fine-tuned encoder. Each one is
> built, each one is tested, and each one is disabled with the p-value that decided it sitting
> next to the switch.
>
> They are off because I measured them and could not show they helped. Shipping them on
> anyway would have been the easy version of this project.

---

## 3:05 – 3:15 · Close

**On screen:** the GitHub repository.

> Code, every measurement, and the ones that came out negative — the link is below.

---

# The post copy

## LinkedIn

> Most RAG demos answer a question. I wanted to build one that could tell you *when it was
> wrong, and which part of it was responsible.*
>
> Warrant answers questions about US federal HR regulation — for a given date, and a given
> scope. The same question asked about 2019 and 2021 has two different correct answers,
> because the regulation was amended in between. It gets both right, and says which is which.
>
> Three things I would point at:
>
> **The citations reach the screen in about half a second; the written answer takes 6.6s.**
> They stream separately, because for someone reading regulation the citation *is* the answer.
> (Retrieval itself is 18ms warm — I quote both, because the small number is the one everyone
> posts and the large one is the one you actually wait for.)
>
> **Every failure is attributable to a stage.** Not "the model hallucinated" — which of
> ingestion, retrieval, fusion, or generation lost the evidence. There is a screen for it.
>
> **Five components ship switched off**, each with the p-value that decided it. A reranker
> worth +0.5 points at p=0.79. An entailment verifier at p=0.55. A fine-tuned encoder that
> moved 8 items out of 383. They are built and measured and disabled, because I could not show
> they helped.
>
> Along the way it caught four of its own mistakes, including three published throughput
> numbers that were wrong by 37%, and 6% of its own citation addresses being malformed — found
> by checking against cross-references the regulation's drafters wrote themselves.
>
> Code and every measurement, including the negative ones:
> github.com/tasnimuldatascience/warrant

## YouTube description

> Warrant is a retrieval system over US federal employment regulation that answers **as of a
> date** — and localizes every wrong answer to the pipeline stage responsible.
>
> 0:00 The same question, two dates, two correct answers
> 0:40 Why the citations arrive before the prose
> 1:15 Follow-ups that cannot drift to a different date
> 1:45 Every version of a section, and which one was law
> 2:10 Which stage is answerable for the answer
> 2:45 The five components that ship switched off
>
> Built on the eCFR point-in-time API. Bitemporal SQLite store, BM25 + dense retrieval fused by
> reciprocal rank, predicates pushed into the query rather than filtered after it. 896 tests,
> 21 measurement write-ups including the negative results.
>
> github.com/tasnimuldatascience/warrant

---

## Notes on delivery

Say the numbers, and say which clock. Retrieval is 18ms warm and about 290ms on the first
query of a fresh process, and a browser sees evidence at roughly half a second once HTTP and
rendering are counted. All three are true and they measure different things; quoting the
smallest as if it were what a user experiences is exactly the error this repository had to
correct on its own throughput figures.

Run `python scripts/demo.py` before filming. It prints both clocks side by side, so you will
say the right one.

Do not apologise for what is missing. If someone asks in the comments whether it scales, the
answer is that the scale study is synthetic, it is written up, and the first thing to break is
the lexical stage at roughly three times the corpus — which is a better answer than most
demos can give.

The five-switched-off slide is the one that gets a reply from an engineer. Give it its time.

---

---

## What is already recorded

`media/` holds a real capture of the interface, made by `scripts/record_ui.py`:

| file | what it is |
|---|---|
| `warrant-demo.mp4` | 1080p, 74s, captions burned in |
| `warrant-demo.gif` | 14s, full frame — general purpose |
| `warrant-linkedin.gif` | 14s, 1000×730, 2.9 MB — cropped to the content column |
| `warrant-architecture.gif` | 13s, 1000×750, **1.4 MB** — the request path, animated |
| `warrant-architecture.mp4` | the same, 1080p, 1.0 MB |
| `architecture.html` | its source; re-record by pointing Playwright at it |
| `warrant-demo.srt` | the same captions, as a sidecar for YouTube |
| `thumbnail.png` | 1280×720, drawn by `scripts/thumbnail.py` |

**The page drives itself.** `#/ask/play` runs the demonstration against the real API and waits
on the actual stream — it is not a staged transcript and not a video loop, so the thing in the
video cannot drift from the thing that runs. Re-record any time with:

```bash
make serve                       # then ask one question, so the model is warm
python scripts/record_ui.py      # writes media/ui-raw.webm
```

Recorded with Playwright rather than a screen grabber, which matters for two reasons: it
captures the *page* and so cannot film your tabs, bookmarks or wallpaper, and the viewport is
fixed, so the framing is identical every time. The first attempt used desktop capture and put
a personal desktop in frame.

**There is no voice-over, deliberately.** Record it separately against the SRT and mux it.
Narrating live ties the pacing of your sentences to the pacing of a language model, and only
one of those is under your control. Captions are burned in because most feed viewers watch
muted.

The thumbnail is drawn rather than screenshotted: a downscaled UI screenshot is unreadable at
the size a feed renders it, which is why most engineering thumbnails are.

### On the LinkedIn GIF specifically

`warrant-linkedin.gif` is cropped to the content column and starts below the masthead, because
a feed renders a GIF at roughly 500px wide and the full 1920px frame reduces the body text to
noise at that size. It opens on the grounded answer and runs through the pinned follow-up:
the one beat that shows a claim tied to a specific version of a specific paragraph, and a
second turn that cannot drift off its date.

2.9 MB, under LinkedIn's limit with room to spare. If it ever needs to be smaller, drop the
frame rate before the width — text survives fewer frames far better than it survives fewer
pixels.

An mp4 will always look better than a GIF in that feed, and LinkedIn autoplays video natively.
The GIF exists for the places that will not take one.

### Which one to post

**`warrant-architecture.gif` is the one for a feed.** A screen recording asks a stranger to
read someone else's UI at 500px wide; the architecture animation states the whole system in
one loop — 9,627 paragraphs admitted, 100 through each retriever, 62 fused, 16 cited, two
claims — with the cost beside every stage and the reranker visibly switched off at p = 0.79.

It is also the only asset that shows the thing worth arguing about: **a component that is
built, measured, and disabled.** Nobody scrolls past that.

Keep `warrant-demo.mp4` for the YouTube upload and for anyone who asks to see it working. The
architecture loop earns the click; the walkthrough rewards it.
