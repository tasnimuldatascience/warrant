"""warrant command line.

    warrant corpus survey  -c CONFIG    how much amendment history each part actually has
    warrant corpus fetch   -c CONFIG    download eCFR point-in-time snapshots (cached)
    warrant corpus build   -c CONFIG    ingest cached snapshots into the bitemporal store
    warrant corpus diff    -c CONFIG    classify what changed between consecutive snapshots
    warrant index build    -c CONFIG    embed the store into the dense index
    warrant eval run       -c CONFIG    score every bucket, with ablations
    warrant autopsy run    -c CONFIG    localize failures; print the failure budget

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
from .config import Config
from .corpus.apparatus import text_of
from .corpus.build import build_part
from .corpus.diff import Change, diff_snapshots
from .corpus.ecfr import ECFRClient
from .corpus.parse import parse_sections
from .eval.bench import mine_all
from .eval.run import score
from .index.store import Store
from .retrieve.dense import DenseIndex
from .retrieve.dense import build as build_dense
from .retrieve.hybrid import Retriever

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
corpus_app = typer.Typer(help="Corpus construction.", no_args_is_help=True)
index_app = typer.Typer(help="Index construction.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation.", no_args_is_help=True)
autopsy_app = typer.Typer(help="Failure localization.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
app.add_typer(autopsy_app, name="autopsy")

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
        index = DenseIndex.load(cfg.dense_path)
    reranker = None
    if rerank and cfg.index.rerank.enabled:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(cfg.index.rerank.model)
    return Retriever(
        store=store, dense_index=index, reranker=reranker,
        candidates_lexical=cfg.retrieve.candidates_lexical,
        candidates_dense=cfg.retrieve.candidates_dense,
        rerank_top_k=cfg.retrieve.rerank_top_k, final_k=cfg.retrieve.final_k,
        temporal=temporal, parts_universe=cfg.corpus.parts,
    )


@index_app.command("build")
def index_build(config: ConfigOpt = None) -> None:
    """Embed every believed chunk into the dense index."""
    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        idx = build_dense(store, model_name=cfg.index.dense.model, config_hash=cfg.hash,
                          batch_size=cfg.index.dense.batch_size, progress=True)
        idx.save(cfg.dense_path)
    console.print(f"[bold]{idx.ids.size}[/bold] vectors, dim {idx.vectors.shape[1]}, "
                  f"model {idx.model} -> {cfg.dense_path}")


def _buckets(cfg: Config, store: Store) -> tuple[dict[str, list], str]:
    horizon = _client(cfg).latest_issue_date(cfg.corpus.title)
    return mine_all(store, horizon=horizon, human_path=cfg.human_path), horizon


@eval_app.command("run")
def eval_run(config: ConfigOpt = None,
             ablate: Annotated[bool, typer.Option(help="also run with predicates off")]
             = True) -> None:
    """Score every bucket, and ablate the as-of predicate.

    The ablation is the point. That the filter works is an assertion; what happens without
    it is a measurement.
    """
    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        buckets, horizon = _buckets(cfg, store)
        console.print(f"horizon {horizon}; " + ", ".join(
            f"{k} {len(v)}" for k, v in sorted(buckets.items())))

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

    table = Table(title="paired deltas (section-clustered, same items)", header_style="bold")
    for col in ("removing", "delta", "95% CI", "won", "lost", "p", "verdict"):
        table.add_column(col, justify="left" if col == "removing" else "right")
    for label, variant in variants:
        b = score(variant, items, samples=cfg.eval.bootstrap_samples)
        d = paired_delta(flags_a, [r.satisfied for r in b.results], keys,
                         samples=cfg.eval.bootstrap_samples)
        table.add_row(label, f"{d.delta * 100:+.1f}", str(d.ci), str(d.wins),
                      str(d.losses), f"{d.p_value:.3g}",
                      "carries its weight" if d.significant else "not measurable")
    console.print(table)


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
                    help="score only every Nth item; 0 scores all")] = 0) -> None:
    """Localize every failure to a stage and print the failure budget."""
    cfg = Config.load(config)
    with Store(cfg.store_path) as store:
        buckets, _ = _buckets(cfg, store)
        items = buckets.get(bucket, [])
        if not items:
            console.print(f"[red]no items in bucket {bucket}[/red]")
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
            from .generate.model import Generator

            gen = Generator()
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
                json.dumps(budget.to_dict(bucket=bucket, config_hash=cfg.hash), indent=2),
                encoding="utf-8")
            console.print(f"recorded -> [bold]{path}[/bold]")


if __name__ == "__main__":
    app()
