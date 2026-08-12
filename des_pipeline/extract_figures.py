"""
The figure route: there is no automatic extraction here, by design.

Reading a value off a plot needs a human with WebPlotDigitizer (or similar), so
this module produces a worklist instead of data. Captions often name the primary
sources for the curves they show, e.g.

    "Fig. 8. Surface tension ... : o, [ChCl:Gly(1:2)] [88]; *, [ChCl:U(1:2)] [90]"

so those reference numbers are captured now — they are the provenance you will
need once the plots are digitised.

Fill in the `extracted_csv` column when a figure has been digitised; nothing in
the pipeline writes to it.
"""
from . import xml_utils
from .schema import FigureRow


def _cited_references(caption):
    numbers = []
    for group in xml_utils.CITATION.findall(caption):
        numbers += xml_utils.expand_ref_field(group)
    return sorted(set(numbers))


def parse_figures(figure_elements, review_doi=""):
    """-> list[FigureRow], one per figure, all flagged for human review."""
    rows = []
    for fig in figure_elements:
        caption = xml_utils.text(fig.find("caption"))
        link = fig.find(".//link")
        rows.append(FigureRow(
            figure_id=fig.get("id", ""),
            label=xml_utils.text(fig.find("label")),
            caption=caption,
            cited_ref_numbers=",".join(str(n) for n in _cited_references(caption)),
            # Namespaces are stripped from tags but not attributes, so the href is
            # still called "{http://www.w3.org/1999/xlink}href".
            image_link=xml_utils.attr_endswith(link, "href") or "" if link is not None else "",
            review_doi=review_doi,
        ))
    return rows
