# eval-021 — Fine-tuning the retriever

**Headline: it does not help, and the reason was visible in the training data before a single
step ran.** Fine-tuning `BAAI/bge-small-en-v1.5` on 493 mined triples moves the held-out split
by +0.9 points on the human bucket and +1.3 on temporal, neither distinguishable from zero
(p = 1.000 and 0.375, on 3 and 5 moved items respectively out of 341). The stock encoder
stays.

## The training data, and the flaw in it

Mined from the **dev split only** — the test split is never seen, and this repository splits
by *section* so the two sides of one amendment cannot straddle the boundary.

```
493 triples (amendment=150, sibling=210, lexical=133)
from 114 items over 39 sections, 57 distinct gold paragraphs
53 contradictory queries
```

**53 of 114 items are contradictory**, and that number is the finding. A contradictory query
appears with more than one gold paragraph — the before and after sides of an amendment. The
same question, asked at two dates, has two different correct answers. That is the premise of
the entire system and it is poison for a contrastive objective: the negative in one triple is
the positive in another, so the gradient pulls the same pair apart and together and the two
contributions largely cancel.

`train/mine.py` counts them because its author expected this. The count had never been printed
before now.

The composition compounds it. 210 of 493 negatives are *siblings* — another paragraph of the
same section — which is deliberate, since paragraph-level discrimination is what this
retrieval needs, but a sibling is precisely where a "hard negative" is most likely to be a
second correct answer rather than a wrong one. The genuinely clean signal is closer to the 133
lexical negatives over 57 distinct gold paragraphs.

## Training

| | |
|---|---|
| triples | 493 |
| steps | 62 over 2 epochs, batch 16 |
| loss | 0.7994 → 0.1855 |
| wall time | **11 s** |
| peak VRAM | 2,575 MB |
| re-embedding 13,212 chunks | 24 s |

`max_seq_length` is set to **192**, not the model's 512. The corpus median chunk is 31 tokens
and p90 is 79, so training at 512 pads every batch to a length almost nothing reaches. Left at
the default the same run took over 25 minutes without finishing and held 7.8 GB; at 192 it is
11 seconds and 2.6 GB. That is a property of this corpus's length distribution, and it is the
single thing most worth knowing before fine-tuning on regulation.

The loss falling from 0.80 to 0.19 is worth reading correctly: **the model learned the training
set.** With 57 distinct gold paragraphs that is close to memorisation, and it is not evidence
of anything about the held-out split — which is why the table below exists.

## Held-out result, paired and per bucket

Test split, dense retrieval only (reranking off so the comparison is of the encoder),
section-clustered intervals, paired — every item scored under both encoders, so only the items
they disagree on carry information.

| bucket | n | stock | tuned | delta | 95% CI | won / lost | p |
|---|---:|---:|---:|---:|:---:|---:|---:|
| human | 108 | 73.1% | 74.1% | **+0.9** | −4.1 – 2.2 | 2 / 1 | 1.000 |
| scope | 42 | 100.0% | 100.0% | 0.0 | 0.0 – 0.0 | 0 / 0 | 1.000 |
| temporal | 233 | 95.7% | 97.0% | **+1.3** | −4.1 – 0.7 | 4 / 1 | 0.375 |

Both intervals cross zero. Across 383 scored items, **eight moved in either direction.**

Reported per bucket rather than aggregate on purpose: the failure mode worth catching is an
improvement on the auto-mined temporal bucket, which is mechanically easy, while the 108
hand-written human items — the ones that measure usefulness — do not move. Temporal is where
the larger delta is, and it is also the bucket the amendment triples were mined from. On the
bucket that matters, three items moved.

## Recommendation

**Keep the stock encoder.** The checkpoint is reproducible from `warrant.train` in 35 seconds
end to end, so nothing is lost by not shipping it, and adopting it would cost a re-embedding
of all 13,212 chunks, a new `data/dense`, and an invalidated quality floor —
`results/eval-floor.json` records the model set precisely so a swapped encoder reports as
*incomparable* rather than being silently graded against the old one.

This is the fifth stage measured and left off, after reranking (+0.5, p=0.79), entailment
(+2.3, p=0.55), the calibrated combiner (AURC +0.0019) and multi-hop (−0.88, p=0.25).

## What would be worth trying next

Not more epochs, and not a bigger model. **Fix the mining first.** Keep an amendment negative
only where its query is unique to one side, so the contradictory pairs stop cancelling; that is
a change to `triples_from_items` and a re-count before any training is worth starting. If the
usable triple count after that is still in the hundreds over a few dozen sections, the honest
conclusion is that this corpus is too small to fine-tune on, and the effort belongs in the
benchmark instead — where growing the human set from 29 to 212 items halved a confidence
interval, which is more than this bought.
