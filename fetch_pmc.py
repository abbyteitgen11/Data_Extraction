#!/usr/bin/env python3
"""
Download the open-access DES corpus from PMC as JATS XML.

    python fetch_pmc.py --counts            # the census only, no downloads
    python fetch_pmc.py --limit 200         # a bounded first pass
    python fetch_pmc.py                     # everything (~13,000 papers, ~3 GB)
    python fetch_pmc.py --manifest-only     # re-describe what is already on disk

The search is just the phrase "deep eutectic solvent" and its plural, with PMC's
open-access filter, and no year restriction -- the filtering happens later, over the
manifest, not here. See des_pipeline/pmc.py for why PMC is the only source available
and what the other three routes were measured to do.

Files land in xml/pmc/, NOT xml/. run_pipeline.py globs xml/*.xml non-recursively, so
nothing downloaded here is run until a human moves it. Nothing is extracted, and no
paper enters the graph.
"""
import argparse
import sys

from des_pipeline import config, pmc


def show_counts(session):
    rows = pmc.counts(session)
    print(f"\nPMC, open access, term: {pmc.TERM}\n")
    print(f"  {'scope':<12} {'phrase anywhere':>16} {'in title/abstract':>19}")
    print("  " + "-" * 49)
    for row in rows:
        print(f"  {row['scope']:<12} {row['anywhere']:>16,} {row['title_abstract']:>19,}")
    print("\n  The two columns differ because PMC searches full text including")
    print("  reference lists: most of the 'anywhere' set only cites DES work.")
    print("  Downloads use the 'anywhere' set; pmc_corpus.csv records which one")
    print("  each paper falls into, so narrowing later needs no refetch.")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", action="store_true",
                        help="print the census and exit; makes no downloads")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N papers (for a first pass)")
    parser.add_argument("--since", type=int, default=None,
                        help="earliest publication year (default: no restriction)")
    parser.add_argument("--out", default=None,
                        help=f"where to write (default: {config.PMC_DIR})")
    parser.add_argument("--manifest-only", action="store_true",
                        help="re-describe the files already on disk, downloading nothing")
    parser.add_argument("--no-manifest", action="store_true",
                        help="download only; skip describing (parsing 13k files is slow)")
    args = parser.parse_args(argv)

    session = pmc.session()
    if not config.NCBI_EMAIL:
        sys.exit("NCBI requires an email address on every request. Set NCBI_EMAIL or "
                 "CROSSREF_MAILTO in .env.")

    if args.counts:
        show_counts(session)
        return

    if not args.manifest_only:
        print("searching PMC:")
        pmcids = pmc.search(session, since=args.since, limit=args.limit)
        print(f"  {len(pmcids)} papers to consider")

        print("\ndownloading:")
        written, skipped, failed = pmc.download(session, pmcids, out_dir=args.out)
        print(f"  wrote {len(written)}, skipped {skipped} already on disk, "
              f"{len(failed)} unavailable")
        if failed:
            # Usually PMC's ingest lag on very recent papers rather than a real absence,
            # so these are worth retrying on a later run, not recorded as permanent.
            print(f"  unavailable (retry later): {', '.join(failed[:10])}"
                  f"{'...' if len(failed) > 10 else ''}")

    if not args.no_manifest:
        print("\ndescribing what was downloaded:")
        rows = pmc.build_manifest(out_dir=args.out)
        if rows:
            from collections import Counter

            where = Counter(r["phrase_in"] for r in rows)
            print(f"  {len(rows)} papers described")
            print("  phrase occurs in: " +
                  ", ".join(f"{k} {v}" for k, v in where.most_common()))
            print(f"  with tables: {sum(1 for r in rows if r['n_tables'])}"
                  f" | with a DOI: {sum(1 for r in rows if r['doi'])}"
                  f" | unreadable: {sum(1 for r in rows if r['error'])}")

    print("\ndone. Nothing has been run through the pipeline.")


if __name__ == "__main__":
    main()
