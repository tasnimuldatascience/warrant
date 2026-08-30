"""Contrastive fine-tune of the dense bi-encoder on mined amendment triples.

The retriever ships an off-the-shelf ``BAAI/bge-small-en-v1.5``, which has never seen a CFR
amendment. ``train.mine`` turns the temporal benchmark into (query, gold, near-duplicate)
triples; this module trains on them and writes a checkpoint an index can be built from.

**Loop, not ``Trainer``.** sentence-transformers' ``fit``/``SentenceTransformerTrainer`` path
requires ``datasets``, which is not in this project's ``neural`` extra -- training against an
undeclared dependency would make the checkpoint unreproducible from a clean install, which is
the one thing a reported number may not be. The loop below is the same objective:
``MultipleNegativesRankingLoss`` over three columns, so every row's loss sees its own hard
negative plus every other row's positive and negative as in-batch negatives.

**Measured, on an RTX 5070 Laptop (8.5 GB) with torch 2.11+cu128:**

===========================================  ==========  =============
configuration                                peak VRAM   wall clock
===========================================  ==========  =============
batch 32, 3 columns, fp32, seq 512, 2 epochs   1.35 GB       57 s
===========================================  ==========  =============

1.35 GB of 8 GB is why the loss is plain ``MultipleNegativesRankingLoss`` and not
``CachedMultipleNegativesRankingLoss``: caching exists to decouple batch size from memory by
running the encoder twice, and there is no memory to buy back here -- batch 32 uses 17% of
the card. fp32 for the same reason; autocast would save memory this run does not need and
would add a second source of run-to-run drift. If the corpus grows enough that batch size
becomes memory-bound, the swap is one constructor argument and the number above is the
evidence for when to make it.

**The query side carries ``QUERY_INSTRUCTION``.** BGE is trained asymmetrically and
``dense.encode_query`` prefixes every served query. Training without it optimises a different
input distribution than the one the index is queried with, and nothing downstream would
report an error -- only slightly worse scores.

**Seeding.** Python, NumPy and torch (CPU and CUDA) are seeded, and the batch order is a
function of the seed and the epoch. cuDNN autotuning is pinned off. A run that cannot be
repeated cannot be reported; the sidecar records the seed alongside everything else needed to
reproduce the checkpoint.
"""

from __future__ import annotations

import json
import math
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..retrieve.dense import DEFAULT_MODEL, QUERY_INSTRUCTION
from .mine import DEFAULT_KINDS, Mined, Triple, batches

#: Written beside the checkpoint. An index built from a directory of weights can otherwise
#: say only "some local path" about what produced it, which is exactly the provenance gap
#: ``DenseIndex`` records ``model`` and ``revision`` to close.
SIDECAR = "warrant-train.json"


@dataclass(frozen=True)
class TrainSpec:
    base_model: str = DEFAULT_MODEL
    #: Pins the HuggingFace commit of the *base* weights. Unpinned, the checkpoint's own
    #: provenance is "whatever main was that day", and the sidecar would be recording a lie
    #: of omission.
    revision: str | None = None
    epochs: int = 2
    #: 32 fits with room to spare (see the module docstring). Larger is not obviously better
    #: here: in-batch negatives are the signal, and the mined set is only a few hundred
    #: triples over 59 distinct paragraphs.
    batch_size: int = 32
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 0
    #: The serving encoder truncates at its own configured maximum; training at a different
    #: one would optimise a document the index never contains.
    max_seq_length: int | None = None
    #: Which mined negative kinds to train on. All of them by default -- see ``train.mine``
    #: for why the amendment negatives are the ones worth ablating.
    kinds: tuple[str, ...] = DEFAULT_KINDS
    #: MNRL's inverse temperature. 20.0 is the library default and the value BGE was trained
    #: under; changing it without a measurement is a free parameter nobody chose.
    scale: float = 20.0
    device: str | None = None


@dataclass
class TrainResult:
    out_dir: Path
    triples: int
    steps: int
    epochs: int
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0
    peak_vram_mb: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        first = self.losses[0] if self.losses else float("nan")
        last = self.losses[-1] if self.losses else float("nan")
        return (f"{self.triples} triples, {self.steps} steps over {self.epochs} epochs in "
                f"{self.seconds:.0f}s; loss {first:.4f} -> {last:.4f}; "
                f"peak VRAM {self.peak_vram_mb:.0f} MB; saved to {self.out_dir}")


def seed_everything(seed: int) -> None:
    """Seed every generator that can move a training run, including CUDA's."""
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency of this project
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Autotuning picks a convolution/GEMM algorithm from timings, so it can differ between
    # two runs of identical code on the same card.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _linear_schedule(step: int, *, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    remaining = total - warmup
    return max(0.0, (total - step) / max(1, remaining))


def finetune(mined: Mined, out_dir: Path | str, *, spec: TrainSpec | None = None,
             progress: bool = False) -> TrainResult:
    """Train on ``mined`` and save a SentenceTransformer checkpoint to ``out_dir``.

    Takes the whole ``Mined`` record rather than a bare list of triples so the sidecar can
    record the per-kind counts and the split the triples came from. A checkpoint that cannot
    say what it was trained on cannot be compared against the encoder it replaced.
    """
    import torch
    from sentence_transformers import SentenceTransformer, losses

    spec = spec or TrainSpec()
    triples: list[Triple] = [t for t in mined.triples if t.kind in set(spec.kinds)]
    if not triples:
        raise ValueError(f"no triples of kinds {spec.kinds}; nothing to train on")

    seed_everything(spec.seed)
    device = spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(spec.base_model, revision=spec.revision, device=device)
    if spec.max_seq_length is not None:
        model.max_seq_length = spec.max_seq_length
    loss_fn = losses.MultipleNegativesRankingLoss(model, scale=spec.scale)

    # Re-batched per epoch from ``seed + epoch``: the batcher holds at most one example per
    # paragraph, so a fixed batching would drop the same overflow examples every epoch.
    plan = [batches(triples, batch_size=spec.batch_size, seed=spec.seed + e)
            for e in range(spec.epochs)]
    total_steps = sum(len(p) for p in plan)
    warmup = int(math.ceil(spec.warmup_ratio * total_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _linear_schedule(s, total=total_steps, warmup=warmup))

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.train()
    started = time.perf_counter()
    losses_seen: list[float] = []
    step = 0
    for epoch, epoch_batches in enumerate(plan):
        for batch in epoch_batches:
            columns = [[QUERY_INSTRUCTION + q for q in batch.queries],
                       batch.positives, batch.negatives]
            features = []
            for column in columns:
                tokenized = model.tokenize(column)
                features.append({k: v.to(device) if hasattr(v, "to") else v
                                 for k, v in tokenized.items()})
            # MNRL ignores ``labels``: the positive of row i is at position i by construction.
            loss = loss_fn(features, labels=None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses_seen.append(float(loss.detach()))
            step += 1
            if progress:
                print(f"  epoch {epoch + 1}/{spec.epochs} step {step}/{total_steps} "
                      f"loss {losses_seen[-1]:.4f}", flush=True)
    seconds = time.perf_counter() - started
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2
            if device.startswith("cuda") else 0.0)

    model.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))

    metadata: dict[str, object] = {
        "base_model": spec.base_model,
        "base_revision": spec.revision,
        "query_instruction": QUERY_INSTRUCTION,
        "triples": len(triples),
        "triple_counts": dict(mined.counts),
        "kinds": list(spec.kinds),
        "train_split": "dev",
        "sections": len(mined.sections),
        "contradictory_queries": mined.contradictory_queries,
        "seed": spec.seed,
        "epochs": spec.epochs,
        "steps": step,
        "batch_size": spec.batch_size,
        "lr": spec.lr,
        "warmup_steps": warmup,
        "scale": spec.scale,
        "max_seq_length": model.max_seq_length,
        "device": device,
        "peak_vram_mb": round(peak, 1),
        "seconds": round(seconds, 1),
        "first_loss": losses_seen[0] if losses_seen else None,
        "last_loss": losses_seen[-1] if losses_seen else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (out / SIDECAR).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return TrainResult(out_dir=out, triples=len(triples), steps=step, epochs=spec.epochs,
                       losses=losses_seen, seconds=seconds, peak_vram_mb=peak,
                       metadata=metadata)


def load_metadata(out_dir: Path | str) -> dict[str, object]:
    """The sidecar written beside a checkpoint, so an index can say what produced it."""
    path = Path(out_dir) / SIDECAR
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
