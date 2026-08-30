"""warrant command line.

    warrant corpus survey  -c CONFIG    how much amendment history each part actually has
    warrant corpus fetch   -c CONFIG    download eCFR point-in-time snapshots (cached)
    warrant corpus build   -c CONFIG    ingest cached snapshots into the bitemporal store
    warrant corpus diff    -c CONFIG    classify what changed between consecutive snapshots

Commands appear here only once the code behind them exists. A CLI that advertises a
subcommand which raises NotImplementedError is worse than one that stays quiet about it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from lxml import etree
from rich.console import Console
from rich.table import Table

from .config import Config
from .corpus.apparatus import text_of
from .corpus.build import build_part
from .corpus.diff import Change, diff_snapshots
from .corpus.ecfr import ECFRClient
from .corpus.parse import parse_sections
from .index.store import Store

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
corpus_app = typer.Typer(help="Corpus construction.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")

console = Console()

ConfigOpt = Annotated[Path | None, typer.Option("-c", "--config", help="config YAML")]

#: Short ASCII column names. The stock Windows console renders a wide table badly and
#: mangles wrapped headers, which makes published numbers harder to read than they deserve.
SHORT_KIND = {
    "substantive_localized": "subst",
    "wholesale_rewrite": "whole",
    "editorial": "edit",
    "apparatus_only": "appar",
    "renumbered": "renum",
    "added": "added",
    "removed": "remvd",
}


def _client(cfg: Config) -> ECFRClient:
    return ECFRClient(cache_dir=cfg.cache_path, delay_s=cfg.corpus.request_delay_s)


def _raw_sections(xml: bytes) -> dict[str, str]:
    """Section text with apparatus left in, for detecting apparatus-only churn."""
    root = etree.fromstring(xml)
    out: dict[str, str] = {}
    for div in root.iter("DIV8"):
        if div.get("TYPE") == "SECTION" and div.get("N"):
            out[div.get("N").strip()] = text_of(div, strip=False)
    return out


@corpus_app.command("survey")
def corpus_survey(config: ConfigOpt = None) -> None:
    """Report how many distinct snapshot dates each part actually has.

    Distinct dates, not version rows: /versions is per-section and overstates diffable
    history by roughly an order of magnitude.
    """
    cfg = Config.load(config)
    client = _client(cfg)
    table = Table(title="eCFR point-in-time history", header_style="bold")
    for col, just in (("part", "right"), ("snapshots", "right"),
                      ("first", "right"), ("last", "right"), ("pairs", "right")):
        table.add_column(col, justify=just)
    total_pairs = 0
    for part in cfg.corpus.parts:
        dates = client.version_dates(cfg.corpus.title, part, floor=cfg.corpus.history_floor)
        pairs = max(len(dates) - 1, 0)
        total_pairs += pairs
        table.add_row(part, str(len(dates)), dates[0] if dates else "-",
                      dates[-1] if dates else "-", str(pairs))
    console.print(table)
    console.print(f"[bold]{total_pairs}[/bold] adjacent snapshot pairs across "
                  f"{len(cfg.corpus.parts)} parts")


@corpus_app.command("fetch")
def corpus_fetch(config: ConfigOpt = None) -> None:
    """Download every retrievable snapshot for the configured parts. Cached and resumable."""
    cfg = Config.load(config)
    client = _client(cfg)
    got = unavailable = 0
    for part in cfg.corpus.parts:
        dates = client.version_dates(cfg.corpus.title, part, floor=cfg.corpus.history_floor)
        have = sum(1 for _ in client.snapshots(cfg.corpus.title, part,
                                               floor=cfg.corpus.history_floor))
        got += have
        unavailable += len(dates) - have
        console.print(f"  part {part}: {have}/{len(dates)} snapshots")
    console.print(f"[bold]{got}[/bold] snapshots cached in {cfg.cache_path}"
                  f" ({unavailable} advertised dates had no retrievable text)")


@corpus_app.command("build")
def corpus_build(config: ConfigOpt = None,
                 rebuild: Annotated[bool, typer.Option(help="delete the store first")] = False
                 ) -> None:
    """Ingest cached snapshots into the bitemporal store."""
    cfg = Config.load(config)
    client = _client(cfg)
    path = cfg.store_path
    if rebuild and path.exists():
        path.unlink()
    with Store(path) as store:
        table = Table(title="bitemporal ingest", header_style="bold")
        for col in ("part", "snaps", "versions", "chunks", "closed", "unchanged"):
            table.add_column(col, justify="right")
        for part in cfg.corpus.parts:
            st = build_part(store, client, title=cfg.corpus.title, part=part,
                            floor=cfg.corpus.history_floor, config_hash=cfg.hash)
            table.add_row(part, str(st.snapshots), str(st.versions_inserted),
                          str(st.chunks_inserted), str(st.sections_closed),
                          str(st.unchanged))
        console.print(table)
        console.print(f"[bold]{store.count()}[/bold] chunk versions in {path}")


@corpus_app.command("diff")
def corpus_diff(config: ConfigOpt = None,
                samples: Annotated[int, typer.Option(help="sample diffs to print")] = 0) -> None:
    """Classify every consecutive-snapshot change, and report the discard rate.

    The discard rate is the point: only ``substantive_localized`` changes can ground a
    benchmark question, and a benchmark that silently drops most of its source material is
    making a representativeness claim it has not earned.
    """
    cfg = Config.load(config)
    client = _client(cfg)
    totals: Counter[str] = Counter()
    per_part: dict[str, Counter[str]] = {}
    shown = 0

    for part in cfg.corpus.parts:
        counts: Counter[str] = Counter()
        prev = prev_raw = prev_date = None
        for date, xml in client.snapshots(cfg.corpus.title, part,
                                          floor=cfg.corpus.history_floor):
            secs = {s.identifier: s.text for s in parse_sections(xml)}
            raw = _raw_sections(xml)
            if prev is not None:
                for ch in diff_snapshots(prev, secs, from_date=prev_date, to_date=date,
                                         before_raw=prev_raw, after_raw=raw):
                    counts[ch.kind.value] += 1
                    if shown < samples and ch.kind is Change.SUBSTANTIVE:
                        shown += 1
                        console.print(f"[dim]{ch.identifier} {ch.from_date} -> {ch.to_date} "
                                      f"sim={ch.similarity:.3f} "
                                      f"changed={ch.changed_tokens}[/dim]")
            prev, prev_raw, prev_date = secs, raw, date
        per_part[part] = counts
        totals.update(counts)

    table = Table(title="changed sections between consecutive snapshots", header_style="bold")
    table.add_column("part", justify="right")
    kinds = [c.value for c in Change]
    for k in kinds:
        table.add_column(SHORT_KIND.get(k, k), justify="right")
    for part, c in per_part.items():
        if sum(c.values()):
            table.add_row(part, *[str(c[k]) for k in kinds])
    table.add_section()
    table.add_row("TOTAL", *[str(totals[k]) for k in kinds], style="bold")
    console.print(table)

    text_changed = sum(totals[k] for k in (Change.SUBSTANTIVE.value,
                                           Change.WHOLESALE.value,
                                           Change.EDITORIAL.value))
    usable = totals[Change.SUBSTANTIVE.value]
    pct = usable / text_changed * 100 if text_changed else 0.0
    console.print(f"regulatory text changed in [bold]{text_changed}[/bold] section-pairs; "
                  f"[bold]{usable}[/bold] usable for the temporal benchmark ({pct:.1f}%)")
    console.print(f"apparatus-only churn suppressed: [bold]"
                  f"{totals[Change.APPARATUS_ONLY.value]}[/bold]")


if __name__ == "__main__":
    app()
