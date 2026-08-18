"""
Where each paper's output lives.

One directory per paper, `data/papers/<slug>/<kind>.csv`, plus `read_all(kind)` to
concatenate across papers for the steps that work on the whole corpus.

Partitioning rather than one big appended file, for two reasons. Re-running one paper
out of a thousand rewrites only its own directory, so it is cheap and it cannot
half-finish a shared file. And it is structurally impossible for paper B to overwrite
paper A's rows -- the failure this pipeline actually had, when a missing DOI fallback
filed one paper's data under another's identity.

Every row still carries `Paper_key` as a column, so a concatenation is self-describing
and a mis-filed row is detectable. `write()` asserts that, which is the cheapest
possible guard against the class of bug above.
"""
import pandas as pd

from . import config

# The per-paper outputs. Anything corpus-wide (components, component_properties)
# stays a single file in data/ because it is keyed by chemical, not by paper.
KINDS = (
    "mixtures",            # one row per DES mixture         (was table2_with_dois.csv)
    "measurements",        # one row per measured value      (was measurements_long.csv)
    "references",          # the bibliography
    "figures",             # the manual-digitisation worklist
    "sections_llm",        # prose measurements
    "skipped_rows",        # table rows we could not read
    "tables_unhandled",    # tables with no usable profile
)


def paper_dir(paper):
    """data/papers/<slug>/ -- created on demand."""
    path = config.PAPERS_DIR / paper.slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_for(kind, paper):
    if kind not in KINDS:
        raise ValueError(f"unknown output kind {kind!r}; expected one of {KINDS}")
    return paper_dir(paper) / f"{kind}.csv"


def write(records, kind, paper, model=None):
    """Write one paper's rows for one kind. -> the Path.

    Stamps and then verifies `Paper_key` on every row: a row that does not know which
    paper it came from must never reach the corpus.
    """
    rows = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in records]
    for row in rows:
        row.setdefault("Paper_key", paper.key)
        row.setdefault("Paper_DOI", paper.doi)
        if not row["Paper_key"]:
            raise ValueError(f"a {kind} row has no Paper_key: {list(row)[:6]}")

    columns = None
    if not rows and model is not None:
        columns = ["Paper_key", "Paper_DOI"] + [
            c for c in model.model_fields if c not in ("Paper_key", "Paper_DOI")]

    path = path_for(kind, paper)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    print(f"  wrote {paper.slug}/{path.name}  ({len(rows)} rows)")
    return path


def read_paper(kind, paper):
    path = path_for(kind, paper)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def read_all(kind):
    """Every paper's rows for one kind, concatenated. -> list[dict].

    The corpus-wide view. Used by the steps that are not per-paper: the graph loader,
    component enrichment, the alias suggester.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown output kind {kind!r}; expected one of {KINDS}")
    rows = []
    for path in sorted(config.PAPERS_DIR.glob(f"*/{kind}.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "Paper_key" not in df.columns:
            raise ValueError(f"{path} has no Paper_key column")
        blank = df.Paper_key.isna().sum()
        if blank:
            raise ValueError(f"{path}: {blank} row(s) with no Paper_key")
        rows += df.astype(object).where(pd.notna(df), None).to_dict("records")
    return rows


def papers():
    """Every paper processed so far. -> list[dict] from data/papers.csv."""
    if not config.PAPERS_CSV.exists():
        return []
    df = pd.read_csv(config.PAPERS_CSV)
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def write_papers(paper_list):
    """data/papers.csv -- one row per paper, the corpus index.

    This is what replaces reading a paper's metadata out of the first row of its own
    table: that breaks with two papers, and breaks entirely for a paper with no table.
    """
    existing = {p["Paper_key"]: p for p in papers()}
    for paper in paper_list:
        existing[paper.key] = paper.as_row()
    rows = sorted(existing.values(), key=lambda r: str(r.get("slug") or ""))
    config.PAPERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.PAPERS_CSV, index=False)
    print(f"  wrote {config.PAPERS_CSV.name}  ({len(rows)} paper(s))")
    return config.PAPERS_CSV


def export_combined(kind, path=None):
    """Optional convenience: one flat CSV of every paper's rows, for humans."""
    rows = read_all(kind)
    path = path or (config.DATA / f"all_{kind}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
