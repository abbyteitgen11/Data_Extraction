"""
Look at the parsed XML, work out what parts it has, and hand each part to the
module that knows how to read it.

Routing is by document *structure*, which is deterministic — we never ask an LLM
what kind of thing it is looking at:

    tables                    -> extract_table       (reliable; the bulk of the data)
    figures                   -> extract_figures     (flagged for manual digitisation)
    bibliography              -> extract_references  (Crossref DOI lookup)
    leaf sections             -> extract_text_llm    (prose; least reliable)

Which elements those *are* is the dialect's business, not this module's, so `route`
takes one and asks it. This module only *classifies*. run_pipeline.py does the
dispatch, which keeps each sub-module runnable on its own.
"""
from dataclasses import dataclass, field


@dataclass
class RoutedDocument:
    tables: list = field(default_factory=list)        # dialects.Table
    figures: list = field(default_factory=list)       # dicts
    references: list = field(default_factory=list)    # lxml elements
    sections: list = field(default_factory=list)      # (id, title, text) tuples
    review: dict = field(default_factory=dict)        # the paper's own metadata


def route(root, dialect):
    """Classify every part of the document. No side effects, no network.

    The dialect supplies every part, so this stays a classifier rather than a second
    place where publisher tag names are written down. It used to reach for
    `.//floats/table` itself, which meant a JATS paper routed to nothing at all while
    `dialects.JATS` sat there able to read it.
    """
    return RoutedDocument(
        tables=dialect.tables(root),
        figures=dialect.figures(root),
        references=dialect.references(root),
        sections=dialect.sections(root),
        review=dialect.paper_metadata(root),
    )


def describe(routed):
    """Print an inventory of what was found, so you can eyeball the routing."""
    lines = [
        f"paper:  {routed.review.get('title', '?')}",
        f"        {routed.review.get('doi', '?')}  "
        f"({routed.review.get('journal', '?')}, {routed.review.get('year', '?')})"
        f"{'  [' + routed.review['article_type'] + ']' if routed.review.get('article_type') else ''}",
        "",
        f"{'kind':<12} {'id':<10} {'label / title':<48} size",
        "-" * 84,
    ]
    for t in routed.tables:
        lines.append(f"{'table':<12} {t.id:<10} {t.label:<48} "
                     f"{len(t.rows)} rows x {t.n_columns} cols")
    for f in routed.figures:
        lines.append(f"{'figure':<12} {f['id']:<10} {f['label']:<48} needs human")
    for sid, title, body in routed.sections:
        lines.append(f"{'section':<12} {sid:<10} {title[:47]:<48} {len(body)} chars")
    lines.append(f"{'references':<12} {'':<10} {'bibliography':<48} "
                 f"{len(routed.references)} entries")
    lines += [
        "-" * 84,
        f"tables {len(routed.tables)} | figures {len(routed.figures)} | "
        f"sections {len(routed.sections)} | references {len(routed.references)}",
    ]
    return "\n".join(lines)
