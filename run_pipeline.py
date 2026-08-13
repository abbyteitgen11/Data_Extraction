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

from des_pipeline import config, router, xml_utils

# "text" runs before "components" so the prose-only component names exist by the time
# the PubChem lookup runs. "table" and "text" both read the reference cache, so "refs"
# has to have run at least once.
# "aliases" is a human-review step -- it proposes abbreviation definitions for you to
# confirm by hand -- so it is deliberately not part of an "all" run.
#ALL_STEPS = ["route", "refs", "table", "figures", "text", "aliases", "components", "graph"]
ALL_STEPS = ["components"]
DEFAULT_STEPS = [s for s in ALL_STEPS if s != "aliases"]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default="all",
                        help=f"comma-separated, or 'all'. one of: {', '.join(ALL_STEPS)}")
    parser.add_argument("--xml", default=str(config.XML))
    parser.add_argument("--no-network", action="store_true",
                        help="use cached lookups only; make no HTTP requests")
    parser.add_argument("--limit", type=int, default=None,
                        help="components: stop after N lookups (for smoke tests)")
    parser.add_argument("--nist", action="store_true",
                        help="components: also scrape the NIST WebBook (fragile)")
    parser.add_argument("--all-sections", action="store_true",
                        help="text: send every prose section, not just the six property ones")
    parser.add_argument("--wipe", action="store_true",
                        help="graph: delete the database before loading")
    parser.add_argument("--no-prose", action="store_true",
                        help="graph: skip the prose measurements (they load by default)")
    parser.add_argument("--refresh-llm", action="store_true",
                        help="text: ignore the response cache and re-call the model")
    args = parser.parse_args(argv)

    steps = DEFAULT_STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        parser.error(f"unknown step(s): {', '.join(unknown)}")

    config.DATA.mkdir(parents=True, exist_ok=True)
    network = not args.no_network

    # Every step except "graph" starts from the parsed document.
    routed = None
    if set(steps) - {"graph"}:
        print(f"parsing {args.xml}")
        routed = router.route(xml_utils.load_root(args.xml))

    if "route" in steps:
        print()
        print(router.describe(routed))
        print()

    # --- references -------------------------------------------------------
    reference_map = None
    if {"refs", "table", "text"} & set(steps):
        from des_pipeline import extract_references as refs

        print("references:")
        reference_map = refs.parse_bibliography(routed.references)
        reference_map = refs.resolve_all(reference_map, network=network)
        reference_map = refs.enrich_all(reference_map, network=network)
        routed.review = refs.enrich_review(routed.review, network=network)
        if "refs" in steps:
            xml_utils.write_csv(refs.to_rows(reference_map), config.REFERENCES_CSV)

    # --- tables -----------------------------------------------------------
    if "table" in steps:
        from des_pipeline import extract_table as tables

        print("tables:")
        table2 = tables.find_table(routed.tables)
        if table2 is None:
            sys.exit("  no Table 2 found")
        rows = tables.parse_table2(table2, reference_map, routed.review)
        xml_utils.write_csv([m for m, _ in rows], config.TABLE_CSV)
        xml_utils.write_csv(tables.to_long(rows), config.LONG_CSV)
        leftover = tables.unhandled_tables(routed.tables, table2)
        if leftover:
            xml_utils.write_csv(leftover, config.TABLES_UNHANDLED_CSV)

    # --- figures ----------------------------------------------------------
    if "figures" in steps:
        from des_pipeline import extract_figures as figures

        print("figures:")
        xml_utils.write_csv(
            figures.parse_figures(routed.figures, routed.review.get("doi", "")),
            config.FIGURES_CSV,
        )

    # --- prose ------------------------------------------------------------
    if "text" in steps:
        from des_pipeline import extract_text_llm as text
        from des_pipeline.schema import LLMMeasurement

        print("prose sections:")
        rows = text.run(routed.sections, routed.review.get("doi", ""),
                        reference_map=reference_map,
                        only_property_sections=not args.all_sections,
                        allow_lookup=network, refresh_llm=args.refresh_llm)
        xml_utils.write_csv(rows, config.SECTIONS_LLM_CSV, model=LLMMeasurement)

    # --- abbreviations ----------------------------------------------------
    if "aliases" in steps:
        from des_pipeline import suggest_aliases

        print("abbreviations:")
        suggest_aliases.report(routed.sections, routed.review.get("doi", ""))

    # --- components -------------------------------------------------------
    if "components" in steps:
        from des_pipeline import enrich_components as components

        print("components:")
        names = components.distinct_components()
        extra = components.prose_components()          # names seen only in the prose
        if extra:
            print(f"  {len(extra)} prose-only component(s): {', '.join(extra)}")
        rows = components.enrich_all(names + extra, limit=args.limit,
                                     network=network, use_nist=not args.nist)
        xml_utils.write_csv(rows, config.COMPONENTS_CSV)

    # --- graph ------------------------------------------------------------
    if "graph" in steps:
        from des_pipeline import build_graph

        print("graph:")
        build_graph.build(wipe=args.wipe, include_prose=not args.no_prose)

    print("\ndone.")


if __name__ == "__main__":
    main()
