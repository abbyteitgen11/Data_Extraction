"""
The identity of one paper, and the one string everything else is keyed on.

Every CSV row, every cache path and every Neo4j MERGE carries `Paper.key`. Getting
this wrong is the worst failure the pipeline has: before this module existed,
`review_metadata()` fell back to a hard-coded DOI constant whenever it did not
recognise a paper's format, so a second paper's rows, references and REVIEW_PAPER
edges all silently landed under the first paper's DOI. There is deliberately no
default anywhere here -- a paper with no DOI gets `xml:<slug>`, which is obviously
provisional, rather than borrowing someone else's identity.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


def normalise_doi(value):
    """'https://doi.org/10.1016/J.X' -> '10.1016/j.x'.

    DOIs are case-insensitive but publishers are inconsistent: 13 of one paper's 197
    reference DOIs are mixed case, and one of those is also cited by another paper in
    lower case. Without normalising at every boundary those become two Paper nodes.
    """
    s = str(value or "").strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    s = re.sub(r"^doi:\s*", "", s, flags=re.I)
    return s.lower()


def slug_of(path):
    """A filesystem-safe stem for per-paper directories and cache filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(path).stem)


@dataclass(frozen=True)
class Paper:
    """One source document. `key` is what every downstream row is stamped with."""

    slug: str
    key: str                     # normalised DOI, or "xml:<slug>" when it has none
    doi: str = ""
    title: str = ""
    authors: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    year: str = ""
    dialect: str = ""            # which format reader parsed it
    path: str = ""
    article_type: str = ""       # the publisher's own label, "" when it states none
    is_review: bool = True       # decides REVIEW_PAPER vs REPORTED_IN for its own data

    @property
    def has_doi(self):
        return bool(self.doi)

    @property
    def role(self):
        """'review' or 'primary'.

        A review tabulates other people's measurements, so its data points at the
        studies it cites. A research paper reports its OWN, so it is itself the
        primary source and must not be recorded as a review of nobody.
        """
        return "review" if self.is_review else "primary"

    def as_row(self):
        """One row of data/papers.csv."""
        return {
            "Paper_key": self.key, "Paper_DOI": self.doi, "slug": self.slug,
            "title": self.title, "authors": self.authors, "journal": self.journal,
            "volume": self.volume, "issue": self.issue, "pages": self.pages,
            "year": self.year, "dialect": self.dialect, "path": self.path,
            "article_type": self.article_type, "role": self.role,
        }


# JATS states the article type outright, so no inference is needed for a PMC paper.
# Anything not on this list is treated as primary research, which is the safer default:
# wrongly calling a review "primary" attaches its data to itself, which is visible and
# wrong; wrongly calling research a "review" leaves its data with no source at all.
REVIEW_TYPES = {"review-article", "systematic-review", "book-review"}


def classify_is_review(meta, abstract="", allow_llm=True):
    """Is this a review or primary research? -> bool.

    JATS states it outright, so this is only reached for formats that do not --
    Elsevier's full-text XML has no article-type attribute. The distinction is not
    cosmetic: a review's measurements belong to the studies it cites, a research
    paper's belong to itself, and getting it backwards either invents a citation or
    loses one.

    Title keywords decide the easy cases without a model call; the model is asked only
    when the title is silent, and its answer is cached like every other LLM call.
    """
    article_type = str(meta.get("article_type", "") or "").lower()
    if article_type:
        return article_type in REVIEW_TYPES

    title = str(meta.get("title", "") or "").lower()
    if re.search(r"\b(review|survey|overview|perspective)\b", title):
        return True
    if re.search(r"\b(experimental|synthesis|measurement[s]?|study on|"
                 r"investigation of)\b", title):
        return False
    if not allow_llm:
        return True

    try:
        from .extract_text_llm import cached_call_llm

        prompt = (
            "Is this chemistry paper a REVIEW (it tabulates measurements taken from "
            "other published studies) or PRIMARY research (it reports measurements the "
            "authors made themselves)?\n\n"
            f"TITLE: {meta.get('title', '')}\n"
            f"JOURNAL: {meta.get('journal', '')}\n"
            f"ABSTRACT: {abstract[:1500]}\n\n"
            'Answer as {"is_review": true} or {"is_review": false}, with one sentence '
            'in "reason".')
        schema = {"type": "object",
                  "properties": {"is_review": {"type": "boolean"},
                                 "reason": {"type": "string"}},
                  "required": ["is_review", "reason"]}
        raw, _cached = cached_call_llm(prompt, schema,
                                       section_id=f"paper-type-{slug_of(path_of(meta))}")
        return bool(json.loads(raw).get("is_review", True))
    except Exception as exc:
        print(f"    could not classify paper type ({type(exc).__name__}); "
              f"assuming review")
        return True


def path_of(meta):
    """The document this metadata came from, for cache naming."""
    return str(meta.get("_path", "paper"))


def from_metadata(meta, path, dialect="", is_review=None):
    """Build a Paper from a dialect's metadata dict. Never invents a DOI.

    `is_review` overrides the publisher's label, for formats that state none --
    Elsevier's full-text XML has no article-type attribute, so the caller asks the
    model once and passes the answer in.
    """
    slug = slug_of(path)
    doi = normalise_doi(meta.get("doi"))
    article_type = str(meta.get("article_type", "") or "")
    if is_review is None:
        is_review = article_type.lower() in REVIEW_TYPES if article_type else True
    return Paper(
        slug=slug,
        key=doi or f"xml:{slug}",
        doi=doi,
        title=meta.get("title", ""),
        authors=meta.get("authors", ""),
        journal=meta.get("journal", ""),
        volume=str(meta.get("volume", "") or ""),
        issue=str(meta.get("issue", "") or ""),
        pages=str(meta.get("pages", "") or ""),
        year=str(meta.get("year", "") or ""),
        dialect=dialect,
        path=str(path),
        article_type=article_type,
        is_review=bool(is_review),
    )
