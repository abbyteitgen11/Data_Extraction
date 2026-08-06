
from lxml.etree import _Element
import re

class Reference:
    """
    This class represents a cited reference in a Paper instance.

    The class will be rarely instantiated by direct call to the constructor (__init__), but 
    rather from factory methods such as @classmethod from_xml or from_dict.

    The method to_dict() creates and returns a dictionary representation. This is useful for
    later serialisation of the instance.

    The only required argument is the reference number (the order in which this reference appears
    cited in the paper). Other possible arguments are authors, journal, title, volume, pages and 
    year, but since some of these may not be provided we only pass them as optional kwargs.
    """

    def __init__(self, 
                 *,
                 refno: int,
                 **kwargs
        ) -> None:

        """ Reference constructor. """

        self.refno = refno

        # -------------------------
        # Authors
        # -------------------------

        self.authors = kwargs.get("authors", [])

        # -------------------------
        # Title
        # -------------------------

        self.title = kwargs.get("title", "")

        # -------------------------
        # Journal
        # -------------------------

        self.journal = kwargs.get("journal", "")

        # -------------------------
        # Publication year
        # -------------------------

        self.year = kwargs.get("year", "")

        # -------------------------
        # Publication volume
        # -------------------------

        self.volume = kwargs.get("volume", "")

        # -------------------------
        # first-page
        # -------------------------

        self.first_page = kwargs.get("first_page", "")

        # -------------------------
        # last-page
        # -------------------------

        self.last_page = kwargs.get("last_page", "")

        # -------------------------
        # articleno (some journals use this instead of pages)
        # -------------------------

        self.articleno = kwargs.get("articleno", "")

        # -------------------------
        # doi (normally this is not given, but just in case...)
        # -------------------------

        self.doi = kwargs.get("doi", "")

        # -------------------------
        # citation id (sometimes Elsevier provides this)
        # -------------------------

        self.citation_id = kwargs.get("citation_id", "")

        # -------------------------
        # raw_xml (in case it is provided)
        # -------------------------

        self.raw_xml = kwargs.get("raw_xml", "")

    def to_dict(self) -> dict:

        """ Create a dict representation. """

        return {
           "refno": self.refno,
           "authors": self.authors,
           "title": self.title,
           "journal": self.journal,
           "year": self.year,
           "volume": self.volume,
           "first_page": self.first_page,
           "last_page": self.last_page,
           "articleno": self.articleno,
           "doi": self.doi,
           "citation_id": self.citation_id,
           "raw_xml": self.raw_xml
        }

    @classmethod
    def from_dict(cls, refdict: dict):

        """ Create an instance of Reference from a dictionary. """

        return cls(
            refno = refdict["refno"],
            authors = refdict["authors"],
            journal = refdict["journal"],
            year = refdict["year"],
            volume = refdict["volume"],
            first_page = refdict["first_page"],
            last_page = refdict["last_page"],
            articleno = refdict["articleno"],
            doi = refdict["doi"]
        )

    @classmethod
    def from_jats_xml(cls, element: _Element):

        """ Create an instance of Reference from JATS PubMed xml data. """

        def get_text(xpath_expr: str) -> str:

            result = element.xpath(xpath_expr)

            if not result:
               return ""

            if isinstance(result, list):
               value = result[0]
            else:
               value = result

            if isinstance(value, str):
               return value.strip()

            return " ".join(value.xpath(".//text()")).strip()

        # ----------------------------------
        # Authors
        # ----------------------------------

        authors = []

        for author in element.xpath(
            ".//*[local-name()='name']"
        ):

            given = author.xpath(
                "string(*[local-name()='given-names'])"
            ).strip()

            surname = author.xpath(
                "string(*[local-name()='surname'])"
            ).strip()

            if given or surname:
                authors.append(
                    f"{given} {surname}".strip()
                )

        # ----------------------------------
        # Article title
        # ----------------------------------

        title = get_text(
            "string("
            ".//*[local-name()='article-title']"
            ")"
        )

        # ----------------------------------
        # Journal title
        # ----------------------------------

        journal = get_text(
            "string("
            ".//*[local-name()='source']"
            ")"
        )

        # ----------------------------------
        # Volume
        # ----------------------------------

        volume = get_text(
            "string(.//*[local-name()='volume'])"
        )

        # ----------------------------------
        # Year
        # ----------------------------------

        year = get_text(
            "string(.//*[local-name()='year'])"
        )

        # ----------------------------------
        # Pages
        # ----------------------------------

        first_page = get_text(
            "string(.//*[local-name()='fpage'])"
        )

        last_page = get_text(
            "string(.//*[local-name()='lpage'])"
        )

        # ----------------------------------
        # Article number
        # ----------------------------------

        article_number = get_text(
            "string(.//*[local-name()='article-number'])"
        )

        # ----------------------------------
        # Article doi (if present)
        # ----------------------------------

        doi = element.xpath(
           "string(.//*[local-name()='doi'])"
        ).strip()

        # ----------------------------------
        # Reference id (if given)
        # ----------------------------------

        citation_id = element.xpath(
           "string(.//*[local-name()='ref']/@id)"
        ).strip()

        # ----------------------------------
        # refno (e.g. if [342], refno = 342)
        # ----------------------------------

        label = element.xpath(
           "string(*[local-name()='label'])"
        ).strip()

        # refno = int(re.match("\[(\d*)\]", label)[1])

        return cls(
            refno = label,
            authors=authors,
            title=title,
            journal=journal,
            volume=volume,
            first_page=first_page,
            last_page=last_page,
            article_number=article_number,
            year=year,
            doi=doi,
            citation_id=citation_id,
            raw_xml=element
        )

    @classmethod
    def from_elsevier_xml(cls, element: _Element):

        """ Create an instance of Reference from Elsevier xml data. """

        def get_text(xpath_expr: str) -> str:

            result = element.xpath(xpath_expr)

            if not result:
               return ""

            if isinstance(result, list):
               value = result[0]
            else:
               value = result

            if isinstance(value, str):
               return value.strip()

            return " ".join(value.xpath(".//text()")).strip()

        # ----------------------------------
        # Authors
        # ----------------------------------

        authors = []

        for author in element.xpath(
            ".//*[local-name()='contribution']"
            "//*[local-name()='author']"
        ):

            given = author.xpath(
                "string(*[local-name()='given-name'])"
            ).strip()

            surname = author.xpath(
                "string(*[local-name()='surname'])"
            ).strip()

            if given or surname:
                authors.append(
                    f"{given} {surname}".strip()
                )

        # ----------------------------------
        # Article title
        # ----------------------------------

        title = get_text(
            "string("
            ".//*[local-name()='contribution']"
            "/*[local-name()='title']"
            "/*[local-name()='maintitle']"
            ")"
        )

        # ----------------------------------
        # Journal title
        # ----------------------------------

        journal = get_text(
            "string("
            ".//*[local-name()='host']"
            "//*[local-name()='series']"
            "/*[local-name()='title']"
            "/*[local-name()='maintitle']"
            ")"
        )

        # ----------------------------------
        # Volume
        # ----------------------------------

        volume = get_text(
            "string(.//*[local-name()='volume-nr'])"
        )

        # ----------------------------------
        # Year
        # ----------------------------------

        year = get_text(
            "string(.//*[local-name()='issue']"
            "/*[local-name()='date'])"
        )

        # ----------------------------------
        # Pages
        # ----------------------------------

        first_page = get_text(
            "string(.//*[local-name()='first-page'])"
        )

        last_page = get_text(
            "string(.//*[local-name()='last-page'])"
        )

        # ----------------------------------
        # Article number
        # ----------------------------------

        article_number = get_text(
            "string(.//*[local-name()='article-number'])"
        )

        # ----------------------------------
        # Article doi (if present)
        # ----------------------------------

        doi = element.xpath(
           "string(.//*[local-name()='doi'])"
        ).strip()

        # ----------------------------------
        # Reference id (if given)
        # ----------------------------------

        citation_id = element.xpath(
           "string(.//*[local-name()='bib-reference']/@id)"
        ).strip()

        # ----------------------------------
        # refno (e.g. if [342], refno = 342)
        # ----------------------------------

        label = element.xpath(
           "string(*[local-name()='label'])"
        ).strip()

        refno = int(re.match("\[(\d*)\]", label)[1])

        return cls(
            refno = refno,
            authors=authors,
            title=title,
            journal=journal,
            volume=volume,
            first_page=first_page,
            last_page=last_page,
            article_number=article_number,
            year=year,
            doi=doi,
            citation_id=citation_id,
            raw_xml=element
        )

