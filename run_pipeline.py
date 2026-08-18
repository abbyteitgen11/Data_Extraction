#!/usr/bin/env python3
"""
Driver for the DES extraction pipeline.

    python run_pipeline.py --steps route                     # inventory only
    python run_pipeline.py --steps table,refs --no-network    # ~5 s, uses the cache
    python run_pipeline.py --steps refs                       # Crossref, ~1 min
    python run_pipeline.py --steps figures
    python run_pipeline.py --steps components --limit 20      # PubChem, slow
    python run_pipeline.py --steps text                       # LLM, needs ollama
    python run_pipeline.py --steps graph --wipe               # load Neo4j

Steps run independently so you can iterate on one without re-hitting Crossref or
PubChem. "table" depends on "refs" having run at least once (it reads the cache).
"""
import argparse
import sys
from pathlib import Path

from des_pipeline import (config, dialects, paper as paper_mod, profile_table, router,
                          store, xml_utils)

# Steps that run once per paper. Everything else works on the whole corpus by reading
# store.read_all(), so it must run after all the papers have been processed.
PER_PAPER_STEPS = ("route", "refs", "table", "figures", "text")


def _inputs(args):
    """The XML files to process, honouring --xml and --papers."""
    if args.xml:
        paths = sorted(Path().glob(args.xml)) if any(c in args.xml for c in "*?[") \
            else [Path(args.xml)]
    else:
        paths = sorted(config.XML_FILES.glob(config.XML_GLOB))
    if args.papers:
        wanted = {s.strip() for s in args.papers.split(",")}
        paths = [p for p in paths if paper_mod.slug_of(p) in wanted]
    if not paths:
        sys.exit(f"no XML files to process (looked in {config.XML_FILES}/)")
    return paths


def _run_paper(path, steps, args, network):
    """Everything that happens to one paper. -> the Paper."""
    root = xml_utils.load_root(path)
    dialect = dialects.detect(root)
    pap = paper_mod.from_metadata(dialect.paper_metadata(root), path, dialect.name)
    if not pap.has_doi:
        raise ValueError(f"{path.name}: no DOI in its metadata; refusing to file its "
                         f"data under a guessed identity")

    print(f"\n=== {pap.slug}  {pap.key}  [{dialect.name}] ===")
    print(f"    {pap.title[:78]}")
    routed = router.route(root)

    if "route" in steps:
        print()
        print(router.describe(routed))
        print()

    reference_map = None
    if {"refs", "table", "text"} & set(steps):
        from des_pipeline import extract_references as refs

        print("references:")
        reference_map = refs.parse_bibliography(routed.references)
        reference_map = refs.resolve_all(reference_map, network=network)
        reference_map = refs.enrich_all(reference_map, network=network)
        routed.review = refs.enrich_review(routed.review, network=network)
        if "refs" in steps:
            store.write(refs.to_rows(reference_map, pap.key), "references", pap)

    if "table" in steps:
        from des_pipeline import extract_table as tables

        print("tables:")
        profiles, problems = profile_table.profile_tables(
            dialect.tables(root), pap, refresh=args.refresh_profiles)
        mixtures, measurements, skipped = tables.extract_tables(
            dialect.tables(root), profiles, pap, reference_map)
        store.write(mixtures, "mixtures", pap, model=tables.MixtureRow)
        store.write(measurements, "measurements", pap, model=tables.MeasurementRow)
        if skipped:
            store.write(skipped, "skipped_rows", pap)
        store.write(tables.unhandled(dialect.tables(root), profiles, problems),
                    "tables_unhandled", pap)

    if "figures" in steps:
        from des_pipeline import extract_figures as figures

        print("figures:")
        store.write(figures.parse_figures(routed.figures, pap.key), "figures", pap)

    if "text" in steps:
        from des_pipeline import extract_text_llm as text
        from des_pipeline.schema import LLMMeasurement

        print("prose sections:")
        rows = text.run(routed.sections, pap.key, reference_map=reference_map,
                        only_property_sections=not args.all_sections,
                        allow_lookup=network, refresh_llm=args.refresh_llm)
        store.write(rows, "sections_llm", pap, model=LLMMeasurement)

    return pap

# "text" runs before "components" so the prose-only component names exist by the time
# the PubChem lookup runs. "table" and "text" both read the reference cache, so "refs"
# has to have run at least once.
# "aliases" is a human-review step -- it proposes abbreviation definitions for you to
# confirm by hand -- so it is deliberately not part of an "all" run.
#ALL_STEPS = ["route", "refs", "table", "figures", "text", "aliases",
#             "components", "validate", "graph"]
ALL_STEPS = ["validate"]
DEFAULT_STEPS = [s for s in ALL_STEPS if s != "aliases"]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default="all",
                        help=f"comma-separated, or 'all'. one of: {', '.join(ALL_STEPS)}")
    parser.add_argument("--xml", default=None,
                        help="one XML file or a glob (default: every file in xml/)")
    parser.add_argument("--papers", default=None,
                        help="comma-separated slugs to restrict the run to")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep going when one paper fails, instead of stopping")
    parser.add_argument("--refresh-profiles", action="store_true",
                        help="table: ignore cached table profiles and re-ask the model")
    parser.add_argument("--no-network", action="store_true",
                        help="use cached lookups only; make no HTTP requests")
    parser.add_argument("--limit", type=int, default=None,
                        help="components: stop after N lookups (for smoke tests)")
    parser.add_argument("--no-nist", action="store_true",
                        help="components: skip the NIST WebBook (it is used by default)")
    parser.add_argument("--no-component-llm", action="store_true",
                        help="components: skip the LLM pass over the PubChem text")
    parser.add_argument("--all-sections", action="store_true",
                        help="text: send every prose section, not just the six property ones")
    parser.add_argument("--wipe", action="store_true",
                        help="graph: delete the database before loading")
    parser.add_argument("--no-prose", action="store_true",
                        help="graph: skip the prose measurements (they load by default)")
    parser.add_argument("--refresh-llm", action="store_true",
                        help="text: ignore the response cache and re-call the model")
    parser.add_argument("--review", action="store_true",
                        help="validate: run the interactive spot check")
    parser.add_argument("--sample", type=int, default=20,
                        help="validate: how many rows per paper to spot-check")
    args = parser.parse_args(argv)

    steps = DEFAULT_STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        parser.error(f"unknown step(s): {', '.join(unknown)}")

    config.DATA.mkdir(parents=True, exist_ok=True)
    network = not args.no_network

    per_paper = [s for s in steps if s in PER_PAPER_STEPS]
    corpus = [s for s in steps if s not in PER_PAPER_STEPS]

    papers = []
    if per_paper:
        for path in _inputs(args):
            try:
                papers.append(_run_paper(path, per_paper, args, network))
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"  FAILED {path.name}: {type(exc).__name__}: {exc}")
        if papers:
            store.write_papers(papers)

    # --- components -------------------------------------------------------
    if "components" in steps:
        from des_pipeline import enrich_components as components

        print("components:")
        names = components.distinct_components()
        extra = components.prose_components()          # names seen only in the prose
        if extra:
            print(f"  {len(extra)} prose-only component(s): {', '.join(extra)}")
        rows = components.enrich_all(names + extra, limit=args.limit,
                                     network=network, use_nist=not args.no_nist)
        xml_utils.write_csv(rows, config.COMPONENTS_CSV)

        if not args.no_component_llm:
            from des_pipeline import component_properties
            from des_pipeline.schema import ComponentPropertyRow

            print("component properties:")
            property_rows = component_properties.run(limit=args.limit,
                                                     refresh_llm=args.refresh_llm)
            xml_utils.write_csv(property_rows, config.COMPONENT_PROPERTIES_CSV,
                                model=ComponentPropertyRow)

    # --- validate ---------------------------------------------------------
    if "validate" in steps:
        from des_pipeline import validate

        print("validate:")
        ok = validate.report(interactive=not args.review, sample=args.sample)
        print(f"\n  overall: {'PASS' if ok else 'needs attention'}")

    # --- graph ------------------------------------------------------------
    if "graph" in steps:
        from des_pipeline import build_graph

        print("graph:")
        build_graph.build(wipe=args.wipe, include_prose=not args.no_prose)

    print("\ndone.")


if __name__ == "__main__":
    main()
