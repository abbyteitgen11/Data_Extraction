"""
The publisher-format seam.

Every publisher writes the same information differently: Elsevier uses CALS tables
(`tgroup/row/entry`, spans via `@morerows`) with floats in `<floats>` and metadata in
`<coredata>`; JATS/PubMed uses XHTML tables (`tr/td`, spans via `@rowspan`) with
metadata in `front/article-meta` and different tag names for almost everything.

Rather than sprinkling `findall(".//table") or findall(".//table-wrap")` through the
pipeline, everything downstream works with the two normalised structures below, and
supporting a new publisher means writing one class here.

Elsevier and JATS/PMC are implemented. `detect()` raises on anything else, loudly and
by name, because the alternative -- guessing -- is what previously let one paper's data
be filed under another paper's DOI.

A dialect answers seven questions about a document: what its tables, figures,
references, bibliography, sections and glossary are, and what paper it is. Everything
format-specific lives behind those; nothing downstream reads a publisher's tag names.
"""
import re
from dataclasses import dataclass, field

from . import xml_utils


@dataclass(frozen=True)
class Cell:
    """One table cell, with the things a parser actually needs kept separate."""

    text: str = ""                          # whitespace-normalised, all lines joined
    lines: tuple = ()                       # split on <break/>: a cell can hold 2-3 values
    markers: tuple = ()                     # <sup> texts -- CANDIDATE footnote markers
    ref_ids: tuple = ()                     # xref/@rid, when the format has them

    def __bool__(self):
        return bool(self.text)


@dataclass
class Table:
    """One table, with its grid already rectangular."""

    id: str = ""
    label: str = ""
    caption: str = ""
    footnotes: str = ""                     # the legend, where footnote markers are defined
    header: list = field(default_factory=list)   # list[list[Cell]]
    rows: list = field(default_factory=list)     # list[list[Cell]]
    ragged: list = field(default_factory=list)   # (row index, cell count) that did not fit
    n_columns: int = 0
    element: object = None

    # The markup declared no header rows, so the real header is sitting in `rows`.
    # 78 of 2547 table-wraps in the PMC corpus are like this, and every one of them
    # extracted nothing: `column()` returned [] for every index, so the header echo in
    # profile_table.validate could not match anything the model said. The profiler's
    # `header_row_count` says how many leading rows to promote; profile_table does it.
    header_missing: bool = False
    # The "table" is a picture of a table. Nothing can be read out of it, but it must be
    # REPORTED rather than skipped -- the same treatment a figure gets.
    graphic_only: bool = False

    def column(self, index):
        """Every header cell at one column index, top row first."""
        return [row[index].text for row in self.header if index < len(row)]

    def promote_header(self, n_rows):
        """Move the first `n_rows` data rows into the header. -> the number moved.

        Only for a table whose markup declared no header. Bounded at half the table so a
        wrong `header_row_count` cannot consume the data: a two-row table claiming five
        header rows keeps its rows and stays unprofiled instead.
        """
        if not self.header_missing or n_rows < 1:
            return 0
        n_rows = min(int(n_rows), max(0, len(self.rows) // 2), 3)
        if n_rows < 1:
            return 0
        self.header = self.rows[:n_rows]
        self.rows = self.rows[n_rows:]
        self.header_missing = False
        return n_rows


# ---------- shared grid machinery ----------
def _expand(raw_rows, cells_of, span_of):
    """Turn rows of possibly-spanning cells into a rectangular grid.

    Row spans are the reason this exists: Sadeghi's Table 2 header has `@morerows="1"`
    on six of its ten columns, so the second header row physically contains only the
    four unit cells. Without expansion, "units" line up under the wrong properties.
    """
    pending = {}            # column index -> (Cell, rows still to fill)
    grid, ragged = [], []
    width = 0

    for n, raw in enumerate(raw_rows):
        row, col = [], 0
        cells = list(cells_of(raw))
        for cell in cells:
            while col in pending:                       # a cell from an earlier row
                held, left = pending[col]
                row.append(held)
                pending[col] = (held, left - 1)
                if left - 1 <= 0:
                    del pending[col]
                col += 1
            value, down, across = span_of(cell)
            for _ in range(max(1, across)):
                row.append(value)
                if down > 0:
                    pending[col] = (value, down)
                col += 1
        while col in pending:                           # trailing held cells
            held, left = pending[col]
            row.append(held)
            pending[col] = (held, left - 1)
            if left - 1 <= 0:
                del pending[col]
            col += 1
        width = max(width, len(row))
        grid.append(row)

    for n, row in enumerate(grid):
        if len(row) != width:
            ragged.append((n, len(row)))
        row += [Cell()] * (width - len(row))            # pad, so indexing is always safe
    return grid, width, ragged


def _cell(el, break_tag="break"):
    """Flatten one cell element, keeping what flattening would otherwise destroy."""
    if el is None:
        return Cell()
    # <break/> separates several values in one cell; itertext() would fuse them into
    # "ChChl:MA (1:1)ChChl:GLY (1:2)".
    parts, buffer = [], []
    def walk(node):
        if node.tag == break_tag:
            parts.append("".join(buffer)); buffer.clear()
        else:
            if node.text:
                buffer.append(node.text)
            for child in node:
                walk(child)
        if node.tail:
            buffer.append(node.tail)
    for child in el:
        walk(child)
    if el.text:
        buffer.insert(0, el.text)
    parts.append("".join(buffer))

    lines = tuple(re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip())
    return Cell(
        text=re.sub(r"\s+", " ", " ".join(lines)).strip(),
        lines=lines or ("",),
        markers=tuple(m for m in xml_utils.sups(el) if m),
        ref_ids=tuple(x.get("rid") for x in el.iter("xref") if x.get("rid")),
    )


# ---------- Elsevier ----------
class Elsevier:
    """Elsevier full-text XML: CALS tables, floats, <coredata>."""

    name = "elsevier"

    @staticmethod
    def matches(root):
        return root.find(".//coredata") is not None

    @staticmethod
    def article_type(root):
        """Elsevier's full-text XML states no article type; the caller asks the model."""
        return ""

    @staticmethod
    def paper_metadata(root):
        return {**xml_utils.review_metadata(root), "article_type": ""}

    @staticmethod
    def tables(root):
        out = []
        for el in (root.findall(".//floats/table") or root.findall(".//table")):
            group = el.find(".//tgroup")
            header_rows = group.findall("./thead/row") if group is not None else []
            body_rows = group.findall("./tbody/row") if group is not None else []
            # Same fallback as JATS: a CALS table with no <tbody> keeps its rows
            # directly under <tgroup>.
            if group is not None and not body_rows:
                body_rows = [r for r in group.findall(".//row") if r not in header_rows]

            # CALS column spans are named ranges; resolve the names to indices once.
            names = [c.get("colname") for c in (group.findall("./colspec") if group is not None else [])]
            def span_of(entry):
                down = int(entry.get("morerows") or 0)
                across = 1
                start, end = entry.get("namest"), entry.get("nameend")
                if start and end and start in names and end in names:
                    across = names.index(end) - names.index(start) + 1
                return _cell(entry), down, across

            header, width_h, _ = _expand(header_rows, lambda r: r.findall("entry"), span_of)
            rows, width_b, ragged = _expand(body_rows, lambda r: r.findall("entry"), span_of)
            width = max(width_h, width_b)
            for row in header + rows:                   # one common width
                row += [Cell()] * (width - len(row))

            out.append(Table(
                id=el.get("id", ""),
                label=xml_utils.text(el.find("label")),
                caption=xml_utils.text(el.find("caption")),
                footnotes=" ".join(xml_utils.text(l) for l in el.findall(".//legend")),
                header=header, rows=rows, ragged=ragged, n_columns=width, element=el,
                header_missing=not header_rows,
            ))
        return out

    @staticmethod
    def figures(root):
        out = []
        for el in (root.findall(".//floats/figure") or root.findall(".//figure")):
            link = el.find(".//link")
            out.append({
                "id": el.get("id", ""),
                "label": xml_utils.text(el.find("label")),
                "caption": xml_utils.text(el.find("caption")),
                "image_link": (xml_utils.attr_endswith(link, "href") or "") if link is not None else "",
            })
        return out

    @staticmethod
    def references(root):
        return (root.findall(".//bibliography//bib-reference")
                or root.findall(".//bib-reference"))

    @staticmethod
    def bibliography(elements):
        """-> {reference number: metadata}. Entries without a numeric label are skipped."""
        refs = {}
        for r in elements:
            label = (r.findtext("label", "") or "").strip("[]")
            if not label.isdigit():
                continue
            authors = []
            for a in r.findall(".//author"):
                given = a.findtext("given-name", "") or ""
                surname = a.findtext("surname", "") or ""
                authors.append(f"{given} {surname}".strip())
            # Elsevier's struct-bib nests two <maintitle>s: the article, then the journal.
            titles = [xml_utils.text(t) for t in r.findall(".//maintitle")]
            refs[int(label)] = {
                "num": int(label),
                "id": r.get("id") or "",
                "authors": authors,
                "title": titles[0] if titles else "",
                "journal": titles[1] if len(titles) > 1 else "",
                "year": (r.findtext(".//date", "") or "").strip(),
                "raw": xml_utils.text(r),
                "doi": None,
            }
        return refs

    @staticmethod
    def sections(root):
        """Leaf sections only -- a parent would repeat all of its children's text."""
        body = root.find(".//body")
        out = []
        for sec in (body if body is not None else root).findall(".//section"):
            if sec.findall("section") or sec.find(".//table") is not None:
                continue
            title = sec.find("section-title")
            out.append((sec.get("id") or "", xml_utils.text(title) if title is not None else "",
                        xml_utils.text(sec)))
        return out

    @staticmethod
    def glossary(root):
        return []                                        # Elsevier has no glossary element

    PARAGRAPH_TAG = "para"

    @staticmethod
    def paragraphs(root):
        """The body's paragraphs, for finding what the prose says about a table.

        Paragraphs only, never whole sections: a <sec> can contain the table itself, so
        pulling section text would feed a table's own contents back into the prompt that
        is meant to interpret it.
        """
        body = root.find(".//body")
        scope = body if body is not None else root
        tag = "para" if scope.find(".//para") is not None else "p"
        return [xml_utils.text(el) for el in scope.findall(f".//{tag}")]



def _citation_text(authors, title, journal, year, fields):
    """A readable citation from parsed fields. -> str.

    `xml_utils.text()` concatenates an element's descendants with no separator, which is
    right for a table cell and wrong for a bibliography entry: it turns Sheldon's into
    "SheldonR. A. Green Chem.2017191843". This is what the review queue shows a human
    and what a Crossref bibliographic query falls back to, so it has to read like a
    citation.
    """
    parts = [", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")]
    parts += [p for p in (title, journal, year) if p]
    if fields.get("volume"):
        parts.append(fields["volume"])
    if fields.get("fpage"):
        parts.append(fields["fpage"] + (f"-{fields['lpage']}" if fields.get("lpage") else ""))
    return ", ".join(p for p in parts if p and p.strip())


# ---------- JATS / PubMed Central ----------
class JATS:
    """NLM JATS as PMC serves it: XHTML tables, <front>/<article-meta>, <ref-list>.

    Used by RSC, MDPI and most PMC-deposited publishers. Two things it gives us that
    Elsevier does not, both of which remove guesswork rather than adding it:

      * `article-type` states outright whether a paper is a review or primary research,
        which decides whether its measurements attach to it as a source or as a review.
      * references frequently carry their own DOI, which beats a fuzzy Crossref match.
    """

    name = "jats"

    @staticmethod
    def matches(root):
        if root.tag == "pmc-articleset":
            return True
        meta = root.find(".//processing-meta")
        return (meta is not None and meta.get("tagset-family") == "jats") or \
            root.find(".//front/article-meta") is not None

    @staticmethod
    def _article(root):
        return root.find(".//article") if root.tag == "pmc-articleset" else root

    @staticmethod
    def article_type(root):
        """'review-article' | 'research-article' | ... -- the publisher's own label."""
        article = JATS._article(root)
        return (article.get("article-type") or "") if article is not None else ""

    @staticmethod
    def paper_metadata(root):
        meta = root.find(".//front/article-meta")
        if meta is None:
            return {}
        journal = root.find(".//front/journal-meta")

        def by_type(tag, attr, value):
            for el in meta.findall(tag):
                if el.get(attr) == value:
                    return xml_utils.text(el)
            return ""

        authors = []
        for name in meta.findall('.//contrib[@contrib-type="author"]/name'):
            given = xml_utils.text(name.find("given-names"))
            surname = xml_utils.text(name.find("surname"))
            if given or surname:
                authors.append(f"{given} {surname}".strip())

        # <pub-date> holds day/month/year as siblings, so text() on the parent fuses
        # them into "392025". Only the <year> child is the year.
        year = ""
        for date in meta.findall("pub-date"):
            year = xml_utils.text(date.find("year")) or year
            if date.get("pub-type") in ("epub", "ppub", "collection") and year:
                break

        pages = ""
        first, last = xml_utils.text(meta.find("fpage")), xml_utils.text(meta.find("lpage"))
        if first:
            pages = f"{first}-{last}" if last else first
        else:
            pages = xml_utils.text(meta.find("elocation-id"))

        return {
            "doi": by_type("article-id", "pub-id-type", "doi"),
            "title": xml_utils.text(meta.find("title-group/article-title")),
            "authors": "; ".join(authors),
            # JATS nests this inside <journal-title-group>, so a direct child lookup
            # silently returns nothing.
            "journal": xml_utils.text(journal.find(".//journal-title"))
            if journal is not None else "",
            "volume": xml_utils.text(meta.find("volume")),
            "issue": xml_utils.text(meta.find("issue")),
            "pages": pages,
            "year": year,
            "article_type": JATS.article_type(root),
        }

    @staticmethod
    def tables(root):
        """Every table in the document, including the awkward ones.

        Measured over 800 corpus papers, 2547 table-wraps: 78 declare no <thead>, 20 no
        <tbody>, 25 hold more than one <table>, and 21 hold a picture instead of a
        table. Every one of those used to produce either nothing or silence.
        """
        out = []
        for wrap in root.findall(".//table-wrap"):
            found = wrap.findall(".//table")
            label = xml_utils.text(wrap.find("label"))
            caption = xml_utils.text(wrap.find("caption"))
            foot = wrap.find(".//table-wrap-foot")
            footnotes = xml_utils.text(foot) if foot is not None else ""

            if not found:
                # A picture of a table. Recorded so `unhandled` can list it for
                # digitisation; skipping it silently is the one option with nothing to
                # be said for it.
                out.append(Table(
                    id=wrap.get("id", ""), label=label, caption=caption,
                    footnotes=footnotes, element=wrap,
                    graphic_only=wrap.find(".//graphic") is not None,
                ))
                continue

            for n, table in enumerate(found):
                header_rows = table.findall(".//thead/tr")
                body_rows = table.findall(".//tbody/tr")
                # No <tbody>: the rows are bare <tr> children of <table>. Reading only
                # tbody gave these tables zero rows, so they profiled and extracted
                # nothing without ever looking wrong.
                if not body_rows:
                    body_rows = [tr for tr in table.iter("tr")
                                 if tr not in header_rows]

                def span_of(cell):
                    # XHTML spans say how many cells this one COVERS; _expand wants how
                    # many further rows it still owns, hence the -1.
                    down = int(cell.get("rowspan") or 1) - 1
                    across = int(cell.get("colspan") or 1)
                    return _cell(cell), down, across

                def cells_of(tr):
                    return [c for c in tr if c.tag in ("td", "th")]

                header, width_h, _ = _expand(header_rows, cells_of, span_of)
                rows, width_b, ragged = _expand(body_rows, cells_of, span_of)
                width = max(width_h, width_b)
                for row in header + rows:
                    row += [Cell()] * (width - len(row))

                out.append(Table(
                    # A second table in one wrap would otherwise share the wrap's id, and
                    # Row_id, the profile cache and the fidelity re-read all key on it.
                    id=wrap.get("id", "") + (f".{n + 1}" if n else ""),
                    label=label + (f" ({n + 1})" if n else ""),
                    caption=caption, footnotes=footnotes,
                    header=header, rows=rows, ragged=ragged, n_columns=width,
                    element=wrap, header_missing=not header_rows,
                ))
        return out

    @staticmethod
    def figures(root):
        out = []
        for el in root.findall(".//fig"):
            graphic = el.find(".//graphic")
            out.append({
                "id": el.get("id", ""),
                "label": xml_utils.text(el.find("label")),
                "caption": xml_utils.text(el.find("caption")),
                "image_link": (xml_utils.attr_endswith(graphic, "href") or "")
                if graphic is not None else "",
            })
        return out

    @staticmethod
    def references(root):
        return root.findall(".//ref-list//ref") or root.findall(".//ref")

    @staticmethod
    def bibliography(elements):
        """-> {reference number: metadata}, numbered by position when unlabelled.

        Neither new paper can use the Elsevier rule. Canela-Xandri's 203 <ref>s have no
        <label> at all; Fan's read "1." which is not `.isdigit()`. JATS orders its
        bibliography, so position IS the citation number -- and `id` is kept because an
        in-text <xref rid="cit59"> is an exact link where a number is only a convention.
        """
        refs = {}
        for position, r in enumerate(elements, 1):
            label = re.sub(r"[^\d]", "", (r.findtext("label", "") or ""))
            num = int(label) if label else position

            citation = r.find("element-citation")
            if citation is None:
                citation = r.find("mixed-citation")
            source = citation if citation is not None else r

            authors = []
            for name in source.findall(".//name"):
                given = xml_utils.text(name.find("given-names"))
                surname = xml_utils.text(name.find("surname"))
                if given or surname:
                    authors.append(f"{given} {surname}".strip())

            doi = ""
            for pid in source.findall(".//pub-id"):
                if pid.get("pub-id-type") == "doi":
                    doi = xml_utils.text(pid)
                    break

            # Volume and pages matter more here than they look. RSC's JATS supplies NO
            # <article-title> at all -- 203 of 203 in Canela-Xandri -- so a citation is
            # identified by author, journal, year, volume and first page or not at all.
            # Discarding these left Crossref searching for "Green Chem. 2017".
            fields = {k: xml_utils.text(source.find(k))
                      for k in ("volume", "fpage", "lpage")}

            refs[num] = {
                "num": num,
                "id": r.get("id") or "",
                "authors": authors,
                "title": xml_utils.text(source.find("article-title")),
                "journal": xml_utils.text(source.find("source")),
                "year": xml_utils.text(source.find("year")),
                "volume": fields["volume"],
                "fpage": fields["fpage"],
                "lpage": fields["lpage"],
                # Built from the parsed fields, not itertext(): the raw XML runs the
                # elements together into "SheldonR. A. Green Chem.2017191843", which is
                # what a human sees in the review queue and on the Paper node.
                "raw": _citation_text(authors, xml_utils.text(source.find("article-title")),
                                      xml_utils.text(source.find("source")),
                                      xml_utils.text(source.find("year")), fields),
                "doi": doi or None,
                "doi_source": "xml" if doi else "",
            }
        return refs

    @staticmethod
    def sections(root):
        """Leaf sections only -- a parent would repeat all of its children's text."""
        body = root.find(".//body")
        out = []
        for position, sec in enumerate(
                (body if body is not None else root).findall(".//sec"), 1):
            if sec.findall("sec") or sec.find(".//table-wrap") is not None:
                continue
            title = sec.find("title")
            # RSC's JATS omits @id on every <sec>. An empty id would make several
            # sections share one identifier downstream, so fall back to position.
            out.append((sec.get("id") or f"sec{position:02d}",
                        xml_utils.text(title) if title is not None else "",
                        xml_utils.text(sec)))
        return out

    @staticmethod
    def glossary(root):
        return root.findall(".//glossary//def-item")

    PARAGRAPH_TAG = "p"

    @staticmethod
    def paragraphs(root):
        """The body's paragraphs, for finding what the prose says about a table.

        Paragraphs only, never whole sections: a <sec> can contain the table itself, so
        pulling section text would feed a table's own contents back into the prompt that
        is meant to interpret it.
        """
        body = root.find(".//body")
        scope = body if body is not None else root
        tag = "para" if scope.find(".//para") is not None else "p"
        return [xml_utils.text(el) for el in scope.findall(f".//{tag}")]



DIALECTS = (Elsevier, JATS)


def detect(root):
    """Which reader handles this document. Raises rather than guessing."""
    for dialect in DIALECTS:
        if dialect.matches(root):
            return dialect
    raise ValueError(
        f"unrecognised XML format (root <{root.tag}>). Only "
        f"{'/'.join(d.name for d in DIALECTS)} is implemented. Add a class to "
        f"des_pipeline/dialects.py rather than letting this paper through unlabelled."
    )
