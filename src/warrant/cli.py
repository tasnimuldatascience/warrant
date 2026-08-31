"""warrant command line.

    warrant corpus survey  -c CONFIG    how much amendment history each part actually has
    warrant corpus fetch   -c CONFIG    download eCFR point-in-time snapshots (cached)
    warrant corpus build   -c CONFIG    ingest cached snapshots into the bitemporal store
    warrant corpus ingest  --source X   add statute, notices, guidance or scans
    warrant corpus diff    -c CONFIG    classify what changed between consecutive snapshots
    warrant index build    -c CONFIG    embed the store into the dense index
    warrant eval run       -c CONFIG    score every bucket on a split, with ablations
    warrant eval gate      -c CONFIG    fail if quality regressed below the floor
    warrant eval generation -c CONFIG   hallucination, citation precision, abstention
    warrant eval abstention -c CONFIG   risk-coverage, calibration, ECE
    warrant eval entailment -c CONFIG   NLI vs span alignment on the labelled probe set
    warrant eval latency   -c CONFIG    latency vs quality per configuration
    warrant autopsy run    -c CONFIG    localize failures; print the failure budget
    warrant replay show    TRACE_ID     what happened on one stored request
    warrant replay diff    TRACE_ID     what today pipeline would do with it instead
    warrant serve          -c CONFIG    run the API on :8000

Commands appear here only once the code behind them exists. A CLI that advertises a
subcommand which raises NotImplementedError is worse than one that stays quiet about it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from lxml import etree
from rich.console import Console
from rich.table import Table

from .autopsy import localize as autopsy
from .config import REPO_ROOT, Config
from .corpus.apparatus import text_of
from .corpus.build import build_part
from .corpus.diff import Change, diff_snapshots
from .corpus.ecfr import ECFRClient
from .corpus.ingest import ingest
from .corpus.parse import parse_sections
from .eval import gate
from .eval.bench import LAST_TEMPORAL_DISCARDS, mine_all
from .eval.entailment import STRATA
from .eval.entailment import load as load_entailment
from .eval.entailment import score as score_entailment
from .eval.run import score
from .index.store import Store, now
from .retrieve.dense import DenseIndex, uncovered
from .retrieve.dense import build as build_dense
from .retrieve.hybrid import Retriever
from .retrieve.multihop import MultiHopRetriever
from .sources.base import AUTHORITY_NAMES

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
corpus_app = typer.Typer(help="Corpus construction.", no_args_is_help=True)
index_app = typer.Typer(help="Index construction.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation.", no_args_is_help=True)
autopsy_app = typer.Typer(help="Failure localization.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
replay_app = typer.Typer(help="Stored traces and replay.", no_args_is_help=True)
app.add_typer(autopsy_app, name="autopsy")
app.add_typer(replay_app, name="replay")

# A floor on width, not a fixed width. Rich sizes to the terminal, and at the 80 columns a
# stock Windows console reports it truncates "adversarial" to "adver..." and a confidence
# interval to "80.9-..." -- a results table whose numbers are elided is a results table that
# has to be re-run somewhere else to be read. 100 matches the line length the code is
# written to plus room for the widest results table; a wider terminal still wins.
console = Console(width=max(106, Console().width))

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


def _generator(cfg: Config):
    """The generator, pinned to whatever the config says.

    A bare repo name resolves to HuggingFace ``main`` on the day of the run, so an unpinned
    generator makes a published answer-quality number depend on a repository nobody here
    controls. ``generate.revision`` is the pin; null keeps today behaviour and says so.
    """
    from .generate.model import Generator

    kwargs: dict[str, object] = {}
    if cfg.generate.model:
        kwargs["model_name"] = cfg.generate.model
    if cfg.generate.revision:
        kwargs["revision"] = cfg.generate.revision
    return Generator(**kwargs)


def _client(cfg: Config, *, refresh: bool = False) -> ECFRClient:
    """The eCFR client, configured from the corpus section.

    The index endpoints carry a TTL and the snapshots do not, which is the whole point: a
    snapshot of a part as of 2019-06-01 is immutable history, while titles.json and the
    per-part version lists are exactly the two endpoints that report new amendments. Caching
    all three forever froze the corpus on the day it was first fetched.
    """
    return ECFRClient(
        cache_dir=cfg.cache_path, delay_s=cfg.corpus.request_delay_s,
        index_ttl_hours=cfg.corpus.index_ttl_hours,
        negative_cache_ttl_days=cfg.corpus.negative_cache_ttl_days,
        issue_date_lag_days=cfg.corpus.issue_date_lag_days,
        refresh=refresh,
    )


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
def corpus_fetch(config: ConfigOpt = None,
                 refresh: Annotated[bool, typer.Option(
                     help="re-fetch the version indexes past their TTL")] = False) -> None:
    """Download every retrievable snapshot for the configured parts.

    Cached and resumable. ``--refresh`` forces the two index endpoints past
    their TTL, which is how a suspected-stale corpus is settled without
    deleting files.
    """
    cfg = Config.load(config)
    client = _client(cfg, refresh=refresh)
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
        for suffix in ("", "-wal", "-shm"):
            p = path.with_name(path.name + suffix)
            if p.exists():
                p.unlink()
    with Store(path) as store:
        if not store.is_empty():
            # Ingesting into a non-empty store used to corrupt it silently: every snapshot
            # was re-applied from scratch, duplicating superseded versions and closing
            # in-force ones at their own start date. Refuse rather than append.
            console.print(
                f"[red]{path} already holds {store.count():,} chunk versions.[/red] "
                "Ingest is not incremental; pass --rebuild to start clean.")
            raise typer.Exit(1)
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


#: The non-eCFR sources, by the name they carry on every chunk they write. Constructed
#: lazily: importing `sources.pdf` pulls in PyMuPDF and an OCR engine, and a user ingesting
#: the US Code should not pay for that.
SOURCE_NAMES = ("federal_register", "usc", "opm", "govinfo")


def _source(name: str, cfg: Config):
    """Build one configured source, or explain why it cannot be built.

    Each source is off in the config by default, so the common first failure is not a stack
    trace but a silent no-op: an enabled=false source that yields nothing looks exactly like
    a source whose API returned nothing. The check is here, once, rather than in four
    ``documents()`` implementations that would each have to be trusted to make it.
    """
    if name == "federal_register":
        c = cfg.sources.federal_register
        from .sources.federal_register import FederalRegisterSource

        return c, FederalRegisterSource(
            cache_dir=_under_root(c.cache_dir), cfr_title=cfg.corpus.title,
            cfr_parts=tuple(c.parts), published_since=c.published_since,
            term=c.term, max_documents=c.max_documents,
            delay_s=cfg.corpus.request_delay_s)

    if name == "usc":
        c = cfg.sources.usc
        from .sources.usc import UscConfig, UscSource

        return c, UscSource(config=UscConfig(
            title=c.title, sections=list(c.sections), chapters=list(c.chapters),
            release_point=c.release_point, cache_dir=_under_root(c.cache_dir),
            request_delay_s=cfg.corpus.request_delay_s))

    if name == "opm":
        c = cfg.sources.opm
        from .sources.html import OPM_FACT_SHEETS, HtmlGuidanceSource

        return c, HtmlGuidanceSource(
            cache_dir=_under_root(c.cache_dir),
            urls=tuple(c.urls) if c.urls else OPM_FACT_SHEETS,
            ttl_hours=c.ttl_hours)

    if name == "govinfo":
        c = cfg.sources.govinfo
        from .sources.pdf import PdfRef, PdfSource

        refs = []
        for spec in c.granules:
            package, _, granule = spec.partition("/")
            refs.append(PdfRef(package=package, granule=granule))
        return c, PdfSource(refs=refs, cache_dir=_under_root(c.cache_dir), ocr=c.ocr)

    raise typer.BadParameter(f"unknown source {name!r}; choose from {', '.join(SOURCE_NAMES)}")


def _under_root(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


@corpus_app.command("ingest")
def corpus_ingest(
    source: Annotated[str, typer.Option(help=f"one of {', '.join(SOURCE_NAMES)}")],
    config: ConfigOpt = None,
) -> None:
    """Add a non-eCFR source to the store: statute, notices, guidance or scans.

    Incremental, unlike `corpus build`. The eCFR path derives validity intervals by diffing
    consecutive point-in-time snapshots, which only works if it applies every snapshot from
    scratch -- so it refuses a non-empty store. These sources hand over documents that
    already know their own dates, so re-running is a content-hash comparison and a
    byte-identical page is a no-op.
    """
    cfg = Config.load(config)
    if source not in SOURCE_NAMES:
        raise typer.BadParameter(f"unknown source {source!r}; "
                                 f"choose from {', '.join(SOURCE_NAMES)}")
    conf, src = _source(source, cfg)
    if not conf.enabled:
        console.print(f"[yellow]sources.{source}.enabled is false[/yellow] in the config. "
                      "Set it to true to ingest; every source is off by default so a clone "
                      "builds the P0 corpus without reaching a network it did not ask for.")
        raise typer.Exit(1)

    with Store(cfg.store_path) as store:
        if store.is_empty():
            console.print("[yellow]the store is empty.[/yellow] Run `warrant corpus build` "
                          "first: these sources corroborate the regulation, and ingesting "
                          "them alone would produce a store no benchmark in this repo can "
                          "score.")
        stats = ingest(store, src.documents(), source=src.name,
                       config_hash=cfg.hash)

    table = Table(title=f"{src.name} ingest", header_style="bold")
    for col in ("documents", "unchanged", "units", "closed", "empty", "failed"):
        table.add_column(col, justify="right")
    table.add_row(str(stats.documents), str(stats.documents_unchanged),
                  str(stats.units_inserted), str(stats.versions_closed),
                  str(stats.documents_empty), str(stats.documents_failed))
    console.print(table)
    for failure in stats.failures[:10]:
        console.print(f"  [red]failed[/red] {failure}")
    if stats.documents_failed > len(stats.failures[:10]):
        console.print(f"  ... and {stats.documents_failed - 10} more")
    console.print(f"[bold]{stats.units_inserted}[/bold] units added at authority "
                  f"{src.authority} ({AUTHORITY_NAMES[src.authority]})")
    _report_reachability(cfg, src.name)


def _report_reachability(cfg: Config, source: str) -> None:
    """Say how much of a source a dated query can actually see.

    This exists because the failure it catches is silent. The US Code source stamped
    ``valid_from`` with the OLRC *edition* date -- 2026-07-12 -- so 5 U.S.C. 6304, in force
    since 1966, was admitted by exactly no query in the 2017-2026 corpus window. The ingest
    reported 38 units and success. Retrieval returned nothing, which is indistinguishable
    from a question with no statutory answer.

    So the check is not "did rows land" but "can the corpus reach them", asked at the
    history floor and at the latest date the store knows about. A source visible at neither
    is a bug however many rows it wrote.
    """
    floor = cfg.corpus.history_floor
    with Store(cfg.store_path) as store:
        latest = store.db.execute(
            "SELECT MAX(valid_from) FROM chunk WHERE system_to IS NULL").fetchone()[0]
        total = store.db.execute(
            "SELECT COUNT(*) FROM chunk WHERE source = ? AND system_to IS NULL",
            (source,)).fetchone()[0]
        if not total:
            return
        seen = {}
        for label, date in (("floor " + floor, floor), ("latest " + str(latest), latest)):
            seen[label] = len(store.candidate_ids(valid_date=date, sources=[source]))

    for label, n in seen.items():
        pct = 100.0 * n / total
        colour = "red" if n == 0 else ("yellow" if pct < 50 else "green")
        console.print(f"  reachable at {label}: [{colour}]{n:,}/{total:,} ({pct:.0f}%)[/{colour}]")
    if not any(seen.values()):
        console.print(
            f"[red]no dated query can reach {source}.[/red] Its valid_from is later than "
            "every date in the corpus, so the as-of predicate excludes all of it. Rows "
            "landing is not the same as a corpus being able to see them.")

    if DenseIndex.exists(cfg.dense_path):
        with Store(cfg.store_path) as store:
            missing = uncovered(DenseIndex.load(cfg.dense_path), store)
        if missing:
            console.print(
                f"  [yellow]{missing:,} believed chunks have no vector.[/yellow] "
                "They are still found lexically, so they appear in one of the two rank "
                "lists RRF fuses instead of two and quietly lose to neighbours that are in "
                "both. Run `warrant index build` to close it.")


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


def _retriever(cfg: Config, store: Store, *, dense: bool = True, rerank: bool = True,
               temporal: bool = True) -> Retriever:  # noqa: D401
    index = None
    if dense and cfg.index.dense.enabled and DenseIndex.exists(cfg.dense_path):
        index = DenseIndex.load(cfg.dense_path,
                                expect_model=cfg.index.dense.model)
    reranker = None
    if rerank and cfg.index.rerank.enabled:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(cfg.index.rerank.model)
    # The multi-hop subclass overrides only retrieve(), so eval, autopsy, replay and the API
    # need no branch of their own -- the hop is a retriever that admits more, not a second
    # pipeline they each have to know about.
    cls = MultiHopRetriever if cfg.retrieve.hop_budget > 0 else Retriever
    extra = ({"hop_budget": cfg.retrieve.hop_budget, "hop_depth": cfg.retrieve.hop_depth}
             if cfg.retrieve.hop_budget > 0 else {})
    return cls(
        store=store, dense_index=index, reranker=reranker,
        candidates_lexical=cfg.retrieve.candidates_lexical,
        candidates_dense=cfg.retrieve.candidates_dense,
        rerank_top_k=cfg.retrieve.rerank_top_k, final_k=cfg.retrieve.final_k,
        temporal=temporal, parts_universe=cfg.corpus.parts,
        config_hash=cfg.hash, reranker_model=cfg.index.rerank.model,
        sources=tuple(cfg.retrieve.sources) or None,
        max_authority=cfg.retrieve.max_authority,
        max_document_frequency=cfg.retrieve.max_document_frequency,
        **extra,
    )


@index_app.command("build")
def index_build(config: ConfigOpt = None) -> None:
    """Embed every believed chunk into the dense index."""
    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        idx = build_dense(store, model_name=cfg.index.dense.model, config_hash=cfg.hash,
                          batch_size=cfg.index.dense.batch_size, progress=True,
                          revision=cfg.index.dense.revision)
        idx.save(cfg.dense_path)
    console.print(f"[bold]{idx.ids.size}[/bold] vectors, dim {idx.vectors.shape[1]}, "
                  f"model {idx.model} -> {cfg.dense_path}")


def _buckets(cfg: Config, store: Store) -> tuple[dict[str, list], str]:
    horizon = _client(cfg).latest_issue_date(cfg.corpus.title)
    return mine_all(store, horizon=horizon, human_path=cfg.human_path), horizon


@eval_app.command("run")
def eval_run(config: ConfigOpt = None,
             split: Annotated[str, typer.Option(
                 help="which split to score: test, dev, or all")] = "test",
             ablate: Annotated[bool, typer.Option(help="also run with predicates off")]
             = True) -> None:
    """Score every bucket on a split, and ablate the predicates.

    The ablation is the point. That a filter works is an assertion; what happens without it
    is a measurement.

    Defaults to **test**. Retrieval parameters were chosen by reading the failure budget, so
    reporting on the same items that chose them is selection on the evaluation set -- run the
    budget on ``--split dev`` and report on test. The split is by section, not by item,
    because two sides of one amendment share a query and would otherwise straddle it.
    """
    if split not in ("test", "dev", "all"):
        console.print(f"[red]unknown split {split!r}; use test, dev or all[/red]")
        raise typer.Exit(2)
    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        buckets, horizon = _buckets(cfg, store)
        if split != "all":
            buckets = {k: [i for i in v if i.split == split] for k, v in buckets.items()}
            buckets = {k: v for k, v in buckets.items() if v}
        console.print(f"horizon {horizon} · split [bold]{split}[/bold]; " + ", ".join(
            f"{k} {len(v)}" for k, v in sorted(buckets.items())))
        if LAST_TEMPORAL_DISCARDS:
            console.print("[dim]temporal items discarded at mining: " + ", ".join(
                f"{k} {v}" for k, v in sorted(LAST_TEMPORAL_DISCARDS.items())) + "[/dim]")

        table = Table(title="benchmark buckets", header_style="bold")
        for col in ("bucket", "n", "sections", "sufficiency", "95% CI",
                    "wrong-version", "95% CI"):
            table.add_column(col, justify="left" if col == "bucket" else "right")
        full = _retriever(cfg, store)
        for _name, items in sorted(buckets.items()):
            table.add_row(*score(full, items, samples=cfg.eval.bootstrap_samples).row())
        if ablate:
            table.add_section()
            no_time = _retriever(cfg, store, temporal=False)
            for name in ("temporal",):
                if name in buckets:
                    table.add_row(*score(no_time, buckets[name],
                                         samples=cfg.eval.bootstrap_samples)
                                  .row(label=f"{name} (as-of off)"))
            flat = _retriever(cfg, store)
            flat.parts_universe = []          # applicability predicate disabled
            for name in ("scope", "scope-exclusion"):
                if name in buckets:
                    table.add_row(*score(flat, buckets[name],
                                         samples=cfg.eval.bootstrap_samples)
                                  .row(label=f"{name} (scope off)"))
        console.print(table)
        console.print("[dim]* enforced by construction: the distractor was never admitted "
                      "by the predicates, so the rate restates the query rather than "
                      "measuring the system. Intervals are a section-clustered bootstrap: "
                      "items from one section are not independent trials.[/dim]")

        if ablate and "temporal" in buckets:
            _paired(cfg, store, buckets["temporal"])


@eval_app.command("gate")
def eval_gate(config: ConfigOpt = None,
              split: Annotated[str, typer.Option(help="which split to gate on")] = "test",
              record: Annotated[bool, typer.Option(
                  help="overwrite the floor with this run instead of checking against it")]
              = False,
              floor: Annotated[Path | None, typer.Option(
                  help="floor file; defaults to store.floor in the config")] = None,
              lexical: Annotated[bool, typer.Option(
                  help="score without dense or reranking, for a runner with no torch")]
              = False) -> None:
    """Fail if quality has regressed below the recorded floor.

    The floor is the lower end of the reference run's section-clustered bootstrap interval,
    not a hand-picked threshold: sufficiency is 97.8% with a 94.9-100 interval, so a gate at
    97.8% would fail about half of all unchanged runs, and one set a little below it would
    be a number nobody could defend. Passing means "not distinguishable from the reference".

    Exits 1 on a regression, 0 otherwise. A configuration change makes the two runs
    incomparable and is reported as such rather than as a pass.
    """
    cfg = Config.load(config)
    # A separate floor per *what ran*, not per config file. A runner with no torch scores the
    # same config lexical-only, so it needs its own recorded numbers -- and the model set
    # written into each floor is what stops the two being compared against each other.
    path = floor or cfg.floor_path
    with Store(cfg.store_path) as store:
        buckets, horizon = _buckets(cfg, store)
        buckets = {k: [i for i in v if i.split == split] for k, v in buckets.items()}
        buckets = {k: v for k, v in buckets.items() if v}
        retriever = _retriever(cfg, store, dense=not lexical, rerank=not lexical)
        results = {name: score(retriever, items, samples=cfg.eval.bootstrap_samples)
                   for name, items in sorted(buckets.items())}
        # What ran, not what was configured. A runner with no torch loads the same config
        # file lexical-only and hashes identically, so the hash alone would grade a lexical
        # run against a reranked floor and call it a pass.
        models = retriever.model_names()

    if record:
        floor = gate.record(results, config_hash=cfg.hash, split=split,
                            recorded_at=now(), models=models)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(floor.to_json(), encoding="utf-8")
        console.print(f"recorded {len(floor.buckets)} bucket floors to [bold]{path}[/bold] "
                      f"at config {cfg.hash}, horizon {horizon}, models "
                      f"{floor.models or 'lexical only'}")
        console.print(f"  items: {floor.items}")
        for b in floor.buckets:
            lo = "n/a" if b.sufficiency_floor is None else f"{b.sufficiency_floor * 100:.1f}%"
            console.print(f"  {b.bucket:<18} n={b.n:<4} sufficiency floor {lo}")
        return

    if not path.exists():
        console.print(f"[red]no floor at {path}.[/red] Record one with "
                      "`warrant eval gate --record` on a run you are willing to defend.")
        raise typer.Exit(2)

    result = gate.check(gate.Floor.load(path), results, config_hash=cfg.hash,
                        models=models)
    if not result.comparable:
        console.print(f"[yellow]incomparable[/yellow] {result.detail}")
        return
    for note in result.improvements:
        console.print(f"  [green]above floor[/green] {note}")
    for violation in result.violations:
        console.print(f"  [red]REGRESSION[/red] {violation}")
    if result.detail:
        console.print(f"[red]{result.detail}[/red]")
    if result.ok:
        console.print(f"[green]gate passed[/green] against the floor recorded at "
                      f"{gate.Floor.load(path).recorded_at}")
        return
    raise typer.Exit(1)


@eval_app.command("abstention")
def eval_abstention(config: ConfigOpt = None,
                    seed: Annotated[int, typer.Option(help="bootstrap seed")] = 0,
                    target_risk: Annotated[float, typer.Option(
                        help="selective risk to hold the operating point to")] = 0.02
                    ) -> None:
    """Risk-coverage for the abstention policy, fitted on dev and reported on test.

    The measurement that motivated this: the generator answered 6 of 29 held-out questions
    with no sufficient evidence retrieved, and abstained on none of them. A system that
    always answers is most dangerous exactly where it is least competent.

    Reports the learned combiner against two baselines it has to beat honestly -- always
    answer, and a threshold on the raw top-1 fusion score. It does not beat the second one,
    which is the finding rather than a failure; see results/eval-005-abstention.md.
    """
    from .verify.calibrate import collect_examples, study

    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        buckets, horizon = _buckets(cfg, store)
        items = [i for v in buckets.values() for i in v]
        examples = collect_examples(_retriever(cfg, store), items,
                                    top_k=cfg.retrieve.final_k)

    dev = [e for e in examples if e.split == "dev"]
    test = [e for e in examples if e.split == "test"]
    if not dev or not test:
        console.print("[red]both splits must be non-empty; nothing to fit or report on[/red]")
        raise typer.Exit(2)
    result = study(dev, test, seed=seed, target_risk=target_risk)

    console.print(f"horizon {horizon} - dev {len(dev)} - test {len(test)} - "
                  f"target selective risk {target_risk:.0%}")
    table = Table(title="risk-coverage on the held-out split", header_style="bold")
    for col in ("policy", "AURC", "95% CI", "coverage", "selective risk"):
        table.add_column(col, justify="left" if col == "policy" else "right")
    for label, policy in (("always answer", result.always_answer),
                          ("top-1 fusion score", result.baseline_top_score),
                          ("learned + isotonic", result.learned)):
        point = policy.at_target
        # AURC as a percentage because its interval is rendered as one. Two adjacent
        # columns in different units is the kind of table a reader silently misreads.
        table.add_row(label, f"{policy.aurc * 100:.2f}%", str(policy.aurc_ci),
                      f"{point.coverage * 100:.1f}%" if point else "-",
                      f"{point.risk * 100:.2f}%" if point else "-")
    console.print(table)
    baseline = result.baseline_top_score
    if baseline.at_target is not None and baseline.at_target.coverage == 0.0:
        # Otherwise this reads as a broken baseline rather than as what it is. The
        # threshold is chosen on dev, where the baseline's best point misses the risk
        # budget by one item and so fails closed -- answering nothing. Choosing it on test
        # instead would hand the baseline a hyperparameter fitted on the reporting set,
        # which is the comparison this repo has already withdrawn a claim for making.
        console.print("[dim]the baseline answers nothing at the shipped threshold: its "
                      "best dev point misses the risk budget by one item, so it fails "
                      "closed. Its threshold is fitted on dev like the combiner's -- "
                      "fitting it on test would be the comparison that flatters it.[/dim]")
    console.print(f"ECE raw [bold]{result.ece_raw:.4f}[/bold] -> "
                  f"calibrated [bold]{result.ece_calibrated:.4f}[/bold]")

    if result.beat_baseline:
        console.print("[green]the learned combiner beats the single-feature baseline[/green]")
    else:
        console.print(
            "[yellow]the learned combiner does not beat a threshold on the top-1 fusion "
            "score.[/yellow] Reported rather than tuned away: eight features, three of them "
            "constant on this corpus, is not enough signal to beat the one that carries it.")


@eval_app.command("entailment")
def eval_entailment(config: ConfigOpt = None,
                    benchmark: Annotated[Path | None, typer.Option(
                        help="benchmark file; defaults to benchmarks/entailment.yaml")] = None,
                    temperature: Annotated[float | None, typer.Option(
                        help="fixed calibration temperature; default is a "
                             "leave-one-section-out fit")] = None,
                    seed: Annotated[int, typer.Option(help="bootstrap seed")] = 0) -> None:
    """Score the entailment verifier against the hand-labelled probe set.

    Two strata, reported separately and never averaged: 129 pairs the generator actually
    emitted, and 53 author-written minimal edits whose class balance was chosen rather than
    observed. Pooling them lets the adversarial contradictions repair a generator stratum
    that detects one contradiction in three.

    Loading fails on an anchor that no longer resolves. That is deliberate: a silently
    skipped pair shrinks the set while the reported n keeps counting what the file says.
    """
    from .verify import entail as entailer_module

    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        pairs = load_entailment(benchmark or cfg.entailment_path, store=store)
    console.print(f"{len(pairs)} pairs over {len({p.section_id for p in pairs})} sections; "
                  + ", ".join(f"{s} {sum(1 for p in pairs if p.stratum == s)}"
                              for s in STRATA))
    report = score_entailment(pairs, temperature=temperature,
                              samples=cfg.eval.bootstrap_samples, seed=seed)

    table = Table(title="entailment accuracy, per stratum", header_style="bold")
    for col in ("stratum", "n", "sections", "micro", "95% CI", "macro",
                "entail", "neutral", "contradict"):
        table.add_column(col, justify="left" if col == "stratum" else "right")
    for s in report.strata:
        table.add_row(*s.row())
    console.print(table)

    compare = Table(title="entailment against span alignment", header_style="bold")
    # Short headers because the row has ten cells: at anything under ~110 columns the
    # verdict cell elides to "measura..." , and a table that hides which comparison is
    # significant is the one cell nobody can afford to lose.
    for col in ("stratum", "n", "agree", "align", "NLI", "delta", "95% CI",
                "won/lost", "p", "verdict"):
        compare.add_column(col, justify="left" if col == "stratum" else "right",
                           no_wrap=col == "verdict")
    for c in report.comparisons:
        compare.add_row(*c.row())
    console.print(compare)

    tp, fn, fp = report.contradiction_rates()
    console.print(f"contradiction recall [bold]{tp / (tp + fn) * 100:.0f}%[/bold] "
                  f"({tp}/{tp + fn}), precision "
                  f"[bold]{tp / (tp + fp) * 100:.0f}%[/bold] ({tp}/{tp + fp}), "
                  f"{fp} false flags")
    console.print("[dim]The two strata are never averaged: the adversarial class balance is "
                  "chosen, not observed. Intervals are section-clustered -- 182 pairs come "
                  "from 91 sections and one over-cited claim contributed ten of them. "
                  "Temperature is " + (f"fixed at {temperature}" if temperature is not None
                                       else "fitted leave-one-section-out") + "; the shipped "
                  f"constant is {entailer_module.CALIBRATION_TEMPERATURE}, and the delta "
                  "moves with it.[/dim]")


def _paired(cfg: Config, store: Store, items: list) -> None:
    """Paired, section-clustered deltas for the comparisons the README makes.

    Reading two marginal intervals for overlap throws away the pairing and is far less
    sensitive; every configuration is scored on identical items, so only the items they
    disagree on carry information.
    """
    from .eval.stats import paired_delta

    base = _retriever(cfg, store)
    variants = [
        ("as-of predicate", _retriever(cfg, store, temporal=False)),
        ("cross-encoder", _retriever(cfg, store, rerank=False)),
    ]
    a = score(base, items, samples=cfg.eval.bootstrap_samples)
    keys = [r.section_id or r.item_id for r in a.results]
    flags_a = [r.satisfied for r in a.results]

    clean_a = [not r.leaked for r in a.results]

    table = Table(title="paired deltas (section-clustered, same items)", header_style="bold")
    for col in ("removing", "measure", "delta", "95% CI", "won", "lost", "p", "verdict"):
        table.add_column(col, justify="left" if col in ("removing", "measure", "verdict")
                         else "right")
    for label, variant in variants:
        b = score(variant, items, samples=cfg.eval.bootstrap_samples)
        # Both measures, because they answer different questions and a predicate can be
        # decisive on one while invisible on the other. With final_k=16 several versions of
        # a section fit in the result list at once, so removing the as-of predicate barely
        # moves sufficiency -- the right paragraph is still in there -- while the wrong
        # version is now in there beside it. Reporting only sufficiency would have called
        # the predicate useless on exactly the bucket built to prove it works.
        for measure, fa, fb in (
            ("sufficiency", flags_a, [r.satisfied for r in b.results]),
            ("no wrong version", clean_a, [not r.leaked for r in b.results]),
        ):
            d = paired_delta(fa, fb, keys, samples=cfg.eval.bootstrap_samples)
            table.add_row(label, measure, f"{d.delta * 100:+.1f}", str(d.ci), str(d.wins),
                          str(d.losses), f"{d.p_value:.3g}",
                          "carries its weight" if d.significant else "not measurable")
        table.add_section()
    console.print(table)


@eval_app.command("generation")
def eval_generation(config: ConfigOpt = None,
                    split: Annotated[str, typer.Option(help="test, dev or all")] = "test",
                    bucket: Annotated[str, typer.Option(help="bucket to score")] = "human",
                    limit: Annotated[int, typer.Option(
                        help="score only this many items; 0 scores all")] = 0) -> None:
    """Measure the generator: hallucination, citation precision, abstention.

    Retrieval quality and answer quality are different questions, and for a while only the
    first was measured. A run where retrieval is perfect and the model invents every claim
    scored as a clean success on every instrument here.
    """
    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        buckets, horizon = _buckets(cfg, store)
        items = buckets.get(bucket, [])
        if split != "all":
            items = [i for i in items if i.split == split]
        if limit:
            stride = max(1, len(items) // limit)
            items = items[::stride][:limit]
        if not items:
            console.print(f"[red]no items in bucket {bucket} / split {split}[/red]")
            raise typer.Exit(1)

        from .eval.generation import score_generation

        console.print(f"scoring [bold]{len(items)}[/bold] {bucket} items "
                      f"(split {split}, horizon {horizon}) — generation is slow")
        report = score_generation(_retriever(cfg, store), _generator(cfg), items,
                                  context_k=cfg.retrieve.context_k)

        table = Table(title=f"generation · {bucket} · split {split}", header_style="bold")
        for col, just in (("measure", "left"), ("value", "right"), ("", "left")):
            table.add_column(col, justify=just)
        for name, value, note in report.rows():
            table.add_row(name, value, note)
        console.print(table)
        console.print("[dim]citation precision counts a citation as good only when the "
                      "chunk was in the context and a supporting span can be located in it: "
                      "a reference to something never shown is fabricated, and one whose "
                      "text does not support the claim is unsupported.[/dim]")


@eval_app.command("latency")
def eval_latency(config: ConfigOpt = None,
                 split: Annotated[str, typer.Option(help="test, dev or all")] = "test",
                 bucket: Annotated[str, typer.Option(help="bucket to time")] = "temporal",
                 limit: Annotated[int, typer.Option(help="queries per configuration")] = 60
                 ) -> None:
    """The latency/quality frontier, measured rather than asserted.

    ARCHITECTURE.md section 10 says which stages may be shed under load is decided by
    measurement and not by declaration -- and for a long time there was no measurement to
    decide it from. This produces one: each configuration is timed on the same queries and
    scored on the same items, so the cost of a stage sits next to what it buys.

    Timings are per-stage wall clock from the trace, so the first query of a run pays the
    encoder load and is discarded as a warm-up rather than smeared across the mean.
    """
    import statistics

    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        buckets, _ = _buckets(cfg, store)
        items = [i for i in buckets.get(bucket, []) if split == "all" or i.split == split]
        if not items:
            console.print(f"[red]no items in {bucket}/{split}[/red]")
            raise typer.Exit(1)
        stride = max(1, len(items) // limit)
        sample = items[::stride][:limit]

        configs = [
            ("lexical only", _retriever(cfg, store, dense=False, rerank=False)),
            ("+ dense", _retriever(cfg, store, rerank=False)),
            ("+ cross-encoder", _retriever(cfg, store)),
        ]

        table = Table(title=f"latency vs quality - {bucket}/{split}, "
                            f"{len(sample)} queries", header_style="bold")
        for col, just in (("configuration", "left"), ("p50 ms", "right"),
                          ("p95 ms", "right"), ("sufficiency", "right"),
                          ("wrong version", "right")):
            table.add_column(col, justify=just)

        for label, retriever in configs:
            # Warm up: the first query of a configuration pays the encoder or cross-encoder
            # load, which is a one-time process cost and not a per-request one.
            retriever.retrieve(sample[0].query, as_of=sample[0].as_of, scope=sample[0].scope)
            totals: list[float] = []
            for item in sample:
                trace = retriever.retrieve(item.query, as_of=item.as_of, scope=item.scope)
                totals.append(trace.timings.get("total", 0.0))
            result = score(retriever, sample, samples=cfg.eval.bootstrap_samples)
            totals.sort()
            p50 = statistics.median(totals)
            p95 = totals[min(int(0.95 * len(totals)), len(totals) - 1)]
            table.add_row(label, f"{p50:.1f}", f"{p95:.1f}",
                          f"{result.sufficiency * 100:.1f}%",
                          f"{result.distractor_rate * 100:.1f}%")
        console.print(table)
        console.print("[dim]A stage may be shed under load only where this table shows it "
                      "inside the noise. Declaring which stages are optional without "
                      "measuring them is how a load-shedding policy trades a slow answer "
                      "for a wrong one.[/dim]")

@autopsy_app.command("run")
def autopsy_run(config: ConfigOpt = None,
                bucket: Annotated[str, typer.Option(help="bucket to autopsy")] = "temporal",
                interventional: Annotated[int, typer.Option(
                    help="failures to also localize by oracle substitution")] = 40,
                write: Annotated[bool, typer.Option(
                    help="record the budget to store.budget for the API and UI")] = True,
                generate: Annotated[bool, typer.Option(
                    help="also localize generation and grounding (slow: needs the model)")]
                = False,
                limit: Annotated[int, typer.Option(
                    help="score only every Nth item; 0 scores all")] = 0,
                split: Annotated[str, typer.Option(
                    help="which split to localize: dev, test or all")] = "dev") -> None:
    """Localize every failure to a stage and print the failure budget.

    Defaults to **dev**. The budget is a tuning instrument -- it is read to decide what to
    change next -- so running it on the split the result is reported from is selection on
    the evaluation set. Tune against `--split dev`, report with `eval run --split test`.
    """
    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        buckets, _ = _buckets(cfg, store)
        items = buckets.get(bucket, [])
        if split != "all":
            items = [i for i in items if i.split == split]
        if not items:
            console.print(f"[red]no items in bucket {bucket} / split {split}[/red]")
            raise typer.Exit(1)
        if limit:
            # A deterministic stride, not a head slice: the buckets are sorted by section id,
            # so the first N items would all come from part 300 and the sample would measure
            # one part rather than the corpus.
            stride = max(1, len(items) // limit)
            items = items[::stride][:limit]
            console.print(f"[dim]sampling {len(items)} items on a stride of {stride}[/dim]")

        gen = None
        if generate:
            gen = _generator(cfg)
        budget = autopsy.run(items, _retriever(cfg, store),
                             interventional_sample=interventional,
                             generator=gen, context_k=cfg.retrieve.context_k)

        console.print(f"[bold]{budget.n}[/bold] items in bucket [bold]{bucket}[/bold], "
                      f"[bold]{budget.failures}[/bold] failures "
                      f"({budget.success_rate * 100:.1f}% satisfied)")
        table = Table(title="observational failure budget (first loss)",
                      header_style="bold")
        for col, just in (("stage", "left"), ("failures", "right"), ("share", "right")):
            table.add_column(col, justify=just)
        for stage, count, share in budget.rows():
            table.add_row(stage, str(count), share)
        console.print(table)

        if budget.repairs:
            rep = Table(title="interventional localization (multi-label; does not sum to N)",
                        header_style="bold")
            rep.add_column("repair", justify="left")
            rep.add_column("implicated", justify="right")
            for name, count in budget.repairs.most_common():
                rep.add_row(name, str(count))
            console.print(rep)
            console.print("[dim]oracle substitution shows a repair works, not that the "
                          "stage was the unique cause[/dim]")

        if write:
            path = cfg.budget_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(budget.to_dict(bucket=f"{bucket}/{split}",
                                          config_hash=cfg.hash), indent=2),
                encoding="utf-8")
            console.print(f"recorded -> [bold]{path}[/bold]")




# -- replay ----------------------------------------------------------------------


def _traces(cfg: Config):
    from .observe.trace_store import TraceStore

    return TraceStore(cfg.traces_path)


@replay_app.command("list")
def replay_list(config: ConfigOpt = None,
                limit: Annotated[int, typer.Option(help="how many to show")] = 20) -> None:
    """Recent stored traces, newest first."""
    cfg = Config.load(config)
    with _traces(cfg) as traces:
        rows = traces.recent(limit=limit)
        if not rows:
            console.print("[dim]no traces recorded yet[/dim]")
            return
        table = Table(title=f"{traces.count()} stored traces", header_style="bold")
        for col in ("trace", "when", "as of", "scope", "cfg", "query"):
            table.add_column(col, justify="left")
        for t in rows:
            table.add_row(t.trace_id, t.created_at[:19], t.as_of, t.scope,
                          (t.config_hash or "-")[:12], t.query[:52])
        console.print(table)


@replay_app.command("show")
def replay_show(trace_id: str, config: ConfigOpt = None) -> None:
    """Artifact replay: exactly what happened on one stored request.

    Reads the trace and nothing else. It does not re-run retrieval, so it still answers
    after the index has been rebuilt -- which is the whole reason it is a separate mode from
    the counterfactual one.
    """
    from .observe.replay import artifact_replay

    cfg = Config.load(config)
    with _traces(cfg) as traces:
        try:
            stored = artifact_replay(traces, trace_id)
        except KeyError:
            console.print(f"[red]no trace {trace_id}[/red]")
            raise typer.Exit(1) from None

        console.print(f"[bold]{stored.query}[/bold]")
        console.print(f"[dim]as of {stored.as_of} - {stored.scope} - "
                      f"cfg {stored.config_hash or '-'} - {stored.created_at[:19]}[/dim]")

        table = Table(title="stages", header_style="bold")
        for col, just in (("stage", "left"), ("n", "right"), ("ms", "right"),
                          ("top of the ranking", "left")):
            table.add_column(col, justify=just)
        # The stage a trace stores and the stage that was timed do not share a name:
        # "fused" is the output of the stage called "fusion". Looking the timing up by the
        # stage name silently rendered every row as "-" -- a missing measurement that looked
        # like a measurement of zero work.
        timing_key = {"fused": "fusion", "reranked": "rerank"}
        for stage in ("lexical", "dense", "fused", "reranked", "final"):
            cands = stored.candidates(stage)
            if not cands:
                continue
            head = ", ".join(
                c.version_id + (f" ({c.score:.3g})" if c.score is not None else "")
                for c in cands[:3])
            ms = stored.timings.get(timing_key.get(stage, stage))
            table.add_row(stage, str(len(cands)), f"{ms:.1f}" if ms else "-", head)
        console.print(table)
        if stored.models:
            console.print("[dim]models: " + ", ".join(
                f"{k}={v}" for k, v in sorted(stored.models.items()) if v) + "[/dim]")


@replay_app.command("diff")
def replay_diff(trace_id: str, config: ConfigOpt = None) -> None:
    """Counterfactual replay: what the current pipeline does with that same request.

    This is the regression harness over real traffic. It re-runs against the vectors and
    chunk boundaries that exist now, whatever the trace says -- corpus text is bitemporally
    replayable, embeddings and chunking are recorded by config hash and never rebuilt -- so a
    diff across a hash change brackets the move without attributing it.
    """
    from .observe.replay import counterfactual_replay

    cfg = Config.load(config)
    with Store(cfg.store_path) as store, _traces(cfg) as traces:
        try:
            d = counterfactual_replay(traces, trace_id, _retriever(cfg, store))
        except KeyError:
            console.print(f"[red]no trace {trace_id}[/red]")
            raise typer.Exit(1) from None

        console.print(f"[bold]{d.query}[/bold]")
        console.print(f"[dim]as of {d.as_of} - {d.scope} - cfg "
                      f"{d.config_hash_then or '-'} -> {d.config_hash_now or '-'}[/dim]")
        if not d.changed:
            console.print("[green]identical at every stage[/green]")
            return

        console.print(f"first divergence: [bold]{d.first_divergence}[/bold]")
        table = Table(title="what moved", header_style="bold")
        for col in ("stage", "entered", "left", "reordered"):
            table.add_column(col, justify="left" if col == "stage" else "right")
        for sd in d.stages:
            if not sd.changed and not sd.reordered:
                continue
            table.add_row(sd.stage, str(len(sd.entered)), str(len(sd.left)),
                          "yes" if sd.reordered else "no")
        console.print(table)
        if d.entered_final or d.left_final:
            console.print(f"[dim]final k: +{len(d.entered_final)} / "
                          f"-{len(d.left_final)}[/dim]")


# -- serve -----------------------------------------------------------------------


@app.command("serve")
def serve(config: ConfigOpt = None,
          host: Annotated[str, typer.Option(help="bind address")] = "127.0.0.1",
          port: Annotated[int, typer.Option(help="port")] = 8000,
          generate: Annotated[bool, typer.Option(help="load the generator")] = True,
          warm: Annotated[bool, typer.Option(
              help="build models at startup rather than on first request")] = True,
          json_logs: Annotated[bool, typer.Option(
              help="structured JSON logs; --no-json-logs for a readable terminal")] = True
          ) -> None:
    """Run the HTTP API.

    Binds to localhost by default. The generation path is capped at roughly three requests
    per minute by an unbatched 1.5B model and admits under a semaphore, refusing with 503 and
    a Retry-After rather than queueing silently -- see serve/api.py for the measurement.
    """
    import uvicorn

    from .observe.logging import configure as configure_logging
    from .serve.api import create_app

    # Before create_app, so the warm-up logs in the chosen format too. Configuring after it
    # would leave exactly the startup lines -- the ones that say why a model failed to
    # load -- in whatever format uvicorn had installed.
    configure_logging(json_output=json_logs)

    cfg = Config.load(config)
    console.print(f"serving [bold]http://{host}:{port}[/bold] - corpus {cfg.store_path.name} "
                  f"- generation {'on' if generate else 'off'}")
    uvicorn.run(create_app(cfg, generate=generate, warm=warm), host=host, port=port,
                log_config=None)  # log_config=None: keep the handler configured above


if __name__ == "__main__":
    app()
