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

    @property
    def has_doi(self):
        return bool(self.doi)

    def as_row(self):
        """One row of data/papers.csv."""
        return {
            "Paper_key": self.key, "Paper_DOI": self.doi, "slug": self.slug,
            "title": self.title, "authors": self.authors, "journal": self.journal,
            "volume": self.volume, "issue": self.issue, "pages": self.pages,
            "year": self.year, "dialect": self.dialect, "path": self.path,
        }


def from_metadata(meta, path, dialect=""):
    """Build a Paper from a dialect's metadata dict. Never invents a DOI."""
    slug = slug_of(path)
    doi = normalise_doi(meta.get("doi"))
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
    )
