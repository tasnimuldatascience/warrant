"""Separating regulatory text from editorial apparatus.

This is the hardest problem in the eCFR ingestion path, and it is not obvious until you
diff two snapshots. eCFR interleaves the regulation itself with editorial material that
changes on publication schedules rather than when the law changes:

  AUTH     authority notes ("Authority: 5 U.S.C. 6311 ...")
  SOURCE   source notes    ("Source: 33 FR 12475, Sept. 4, 1968 ...")
  EDNOTE   editorial notes
  FTNT     footnotes
  EFFDNOT  effective-date notes
  CITA     citation blocks
  XREF     cross-references -- MOST are real regulatory text and must be kept, but the
           pending-amendment pointers are not:

               <XREF ID="20200810" REFID="1">Link to an amendment published at
               85 FR 48089, Aug. 10, 2020.</XREF>

           These appear while an amendment is pending and vanish when it publishes.

In the first spike run against Part 630, six of eight sampled "substantive" amendments were
nothing but those pointers appearing or disappearing. A differ that does not strip them
reports publication-schedule churn as changes in the law.
"""

from __future__ import annotations

import copy
import re

from lxml import etree

#: Elements that are editorial apparatus in their entirety.
APPARATUS_TAGS = frozenset({"AUTH", "SOURCE", "EDNOTE", "FTNT", "EFFDNOT", "CITA"})

#: Pending-amendment pointers, which live inside otherwise-legitimate XREF elements.
AMENDMENT_LINK = re.compile(r"link\s+to\s+an?\s+amendment\s+published\s+at", re.I)

_WS = re.compile(r"\s+")


def is_apparatus(el: etree._Element) -> bool:
    """True if this element is editorial apparatus rather than regulatory text."""
    if el.tag in APPARATUS_TAGS:
        return True
    if el.tag == "XREF":
        return bool(AMENDMENT_LINK.search("".join(el.itertext())))
    return False


def strip_apparatus(node: etree._Element, *, inplace: bool = False) -> etree._Element:
    """Remove apparatus subtrees, preserving the prose that surrounds them.

    Tail text matters: an apparatus element sitting mid-paragraph carries the rest of the
    sentence in its ``.tail``. Dropping the element without reattaching the tail silently
    deletes regulatory text -- a bug that shows up much later as an unexplained retrieval
    miss, which is exactly the kind of failure this project exists to attribute.

    Idempotent by construction: a second call finds nothing left to remove.
    """
    root = node if inplace else copy.deepcopy(node)
    for el in list(root.iter()):
        if el is root or not is_apparatus(el):
            continue
        parent = el.getparent()
        if parent is None:
            continue
        if el.tail:
            prev = el.getprevious()
            if prev is not None:
                prev.tail = (prev.tail or "") + el.tail
            else:
                parent.text = (parent.text or "") + el.tail
        parent.remove(el)
    return root


def text_of(node: etree._Element, *, strip: bool = True) -> str:
    """Normalised text content of a node, with apparatus removed by default."""
    target = strip_apparatus(node) if strip else node
    return _WS.sub(" ", "".join(target.itertext())).strip()
