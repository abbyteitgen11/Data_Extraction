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


def parse_figures(figures, review_doi=""):
    """-> list[FigureRow], one per figure, all flagged for human review.

    Takes the dialect's normalised dicts rather than raw elements: Elsevier's
    `<figure><link xlink:href>` and JATS's `<fig><graphic xlink:href>` differ only in
    tag names, and resolving that is `dialects`' job, not this module's.
    """
    rows = []
    for fig in figures:
        caption = fig.get("caption", "")
        rows.append(FigureRow(
            figure_id=fig.get("id", ""),
            label=fig.get("label", ""),
            caption=caption,
            cited_ref_numbers=",".join(str(n) for n in _cited_references(caption)),
            image_link=fig.get("image_link", ""),
            review_doi=review_doi,
        ))
    return rows
