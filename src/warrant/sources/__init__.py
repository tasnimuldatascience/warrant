"""Data sources. Each yields SourceDoc objects; none writes to the store."""

from .base import (
    AUTHORITY_ARCHIVAL,
    AUTHORITY_GUIDANCE,
    AUTHORITY_NAMES,
    AUTHORITY_NOTICE,
    AUTHORITY_REGULATION,
    AUTHORITY_STATUTE,
    KIND_CAPTION,
    KIND_HEADING,
    KIND_OCR,
    KIND_PROSE,
    KIND_TABLE,
    Source,
    SourceDoc,
    Unit,
    merge_anchors,
)

__all__ = [
    "AUTHORITY_ARCHIVAL", "AUTHORITY_GUIDANCE", "AUTHORITY_NAMES", "AUTHORITY_NOTICE",
    "AUTHORITY_REGULATION", "AUTHORITY_STATUTE", "KIND_CAPTION", "KIND_HEADING", "KIND_OCR",
    "KIND_PROSE", "KIND_TABLE", "Source", "SourceDoc", "Unit", "merge_anchors",
]
