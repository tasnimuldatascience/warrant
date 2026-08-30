"""Persisted traces and the two replay modes built on them (ARCHITECTURE.md section 8).

Retrieval writes a `warrant.retrieve.hybrid.Trace` for every request; this package is where
that trace outlives the process. `TraceStore` keeps it in its own SQLite file, and `replay`
reads it back two ways -- exactly, from the record alone, or against today's pipeline as a
regression diff.
"""

from .replay import (
    ReplayDiff,
    StageDiff,
    artifact_replay,
    counterfactual_replay,
    counterfactual_sweep,
    diff,
)
from .trace_store import TRACE_SCHEMA_VERSION, StoredTrace, TraceStore

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "ReplayDiff",
    "StageDiff",
    "StoredTrace",
    "TraceStore",
    "artifact_replay",
    "counterfactual_replay",
    "counterfactual_sweep",
    "diff",
]
