"""
Look at the parsed XML, work out what parts it has, and hand each part to the
module that knows how to read it.

Routing is by document *structure*, which is deterministic — we never ask an LLM
what kind of thing it is looking at:

    floats/table              -> extract_table       (reliable; the bulk of the data)
    floats/figure             -> extract_figures     (flagged for manual digitisation)
    tail/bibliography         -> extract_references  (Crossref DOI lookup)
    body//section (leaves)    -> extract_text_llm    (prose; least reliable)

This module only *classifies*. run_pipeline.py does the dispatch, which keeps each
sub-module runnable on its own.
"""
from dataclasses import dataclass, field

from . import xml_utils


@dataclass
class RoutedDocument:
    tables: list = field(default_factory=list)        # lxml elements
    figures: list = field(default_factory=list)       # lxml elements
    references: list = field(default_factory=list)    # lxml elements
    sections: list = field(default_factory=list)      # (id, title, text) tuples
    review: dict = field(default_factory=dict)        # from <coredata>


def _section_title(section):
    """The section's own title, not a descendant subsection's."""
    own = section.find("section-title")
    return xml_utils.text(own) if own is not None else ""


def route(root):
    """Classify every part of the document. No side effects, no network."""
    # Elsevier puts tables and figures in <floats>, outside the body text, so the
    # container path alone identifies them. Fall back to a document-wide search in
    # case a future paper nests them.
    tables = root.findall(".//floats/table") or root.findall(".//table")
    figures = root.findall(".//floats/figure") or root.findall(".//figure")
    references = root.findall(".//bibliography//bib-reference") or root.findall(
        ".//bib-reference"
    )

    # Leaf sections only. Taking every <section> would emit the parent
    # "Prepared DESs and their physical properties" *and* each of its ten
    # children, sending the same 46k characters to the LLM twice.
    sections = []
    body = root.find(".//body")
    for sec in (body if body is not None else root).findall(".//section"):
        if sec.findall("section") or sec.find(".//table") is not None:
            continue
        sections.append((sec.get("id") or "", _section_title(sec), xml_utils.text(sec)))

    return RoutedDocument(
        tables=tables,
        figures=figures,
        references=references,
        sections=sections,
        review=xml_utils.review_metadata(root),
    )


def describe(routed):
    """Print an inventory of what was found, so you can eyeball the routing."""
    lines = [
        f"review: {routed.review.get('title', '?')}",
        f"        {routed.review.get('doi', '?')}  "
        f"({routed.review.get('journal', '?')}, {routed.review.get('year', '?')})",
        "",
        f"{'kind':<12} {'id':<10} {'label / title':<48} size",
        "-" * 84,
    ]
    for t in routed.tables:
        label = xml_utils.text(t.find("label"))
        lines.append(
            f"{'table':<12} {t.get('id', ''):<10} {label:<48} {len(t.findall('.//row'))} rows"
        )
    for f in routed.figures:
        label = xml_utils.text(f.find("label"))
        lines.append(f"{'figure':<12} {f.get('id', ''):<10} {label:<48} needs human")
    for sid, title, body in routed.sections:
        lines.append(f"{'section':<12} {sid:<10} {title[:47]:<48} {len(body)} chars")
    lines.append(f"{'references':<12} {'':<10} {'bibliography':<48} {len(routed.references)} entries")
    lines += [
        "-" * 84,
        f"tables {len(routed.tables)} | figures {len(routed.figures)} | "
        f"sections {len(routed.sections)} | references {len(routed.references)}",
    ]
    return "\n".join(lines)
