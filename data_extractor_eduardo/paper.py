from lxml import etree

from figure import Figure
from section import Section
from table import Table
from reference import Reference

def first_or_none(lst):
    return lst[0] if lst else None

class PaperException(Exception):
    pass

class Paper:
    """
    This class represents a paper entry for the database. Class data is stored in a series of
    data and data structures. A Paper instance must contain at least a title (which can be empty) 
    and a doi (a unique identifier), but in general an instance will contain a complete 
    representation of the paper (see below) except for the graphical information (though 
    figure captions are also included). 

    title: str
    doi: digital object identifier
    identifiers: dict[str, str] (pmid, pmcid, etc)

    (note that except for doi, which always exists, other identifiers are source dependent, e.g.
    pmid and pmcid are specific to PubMed; PII and EID are specific to Elsevier, etc.)

    The rest of the data in the instance is passed through optional keywords:

    journal: str
    article_type: (experimental, theoretical, review, etc)
    year: str
    authors: list[str]
    abstract: str
    sections: list[dict[title,list[paragraphs]]] (title (str), paragraphs list[str])
    figures: list[Figure] (list of instances of class Figure, see figure.py)
    tables: list[Table] (list of instances of class Table, see table.py)

    Normally the constructor __init__ is not invoqued directly; rather, the @classmethod
    from_xml() is used instead. This facilitates the creation of Paper instances from xml data, 
    which will be the most common source; eventually there may be other construction methods, e.g.
    from html or pdf files.

    to_dict(self) function returns a dictionary with paper data; this is useful for 
    json serialisation 

    paper = {
    "title": "...",
    "doi": "...",
    "identifiers": "...",
    "journal": "...",
    "year": "...",
    "article_type": "...",
    "authors": [...],
    "abstract": "..."
    ...
    }

    The class contains functions to populate the key values from an xml file and a printer function
    for ease of visualisation.

    """

    def __init__(self,
                 *,
                 title: str,
                 doi: str, 
                 **kwargs
                 ) -> None:

        """ 
        Create an instance of Paper 

        Note the presence of '*' before title and doi; this forces title and doi to be 
        keyword arguments; it is done like this since both variables are str, and we want
        to avoid a Paper getting its title as the doi and/or viceversa.
        """

        # -------------------------
        # Title
        # -------------------------

        self.title = title

        # -------------------------
        # DOI
        # -------------------------

        self.doi = doi

        # now check for other provided data

        # -------------------------
        # Identifiers (these may depend on publisher)
        # -------------------------
     
        self.identifiers = kwargs.get("identifiers", {})

        # -------------------------
        # Article type
        # -------------------------

        self.article_type = kwargs.get("article_type", "")

        # -------------------------
        # Journal
        # -------------------------

        self.journal = kwargs.get("journal", "")

        # -------------------------
        # Publication year
        # -------------------------

        self.year = kwargs.get("year", "")

        # -------------------------
        # Abstract
        # -------------------------

        self.abstract = kwargs.get("abstract", "")

        # -------------------------
        # Authors
        # -------------------------

        self.authors = kwargs.get("authors", [])

        # -------------------------
        # Sections and paragraphs
        # -------------------------

        self.sections = kwargs.get("sections", [])

        # -------------------------
        # Figures (captions only)
        # -------------------------

        self.figures = kwargs.get("figures", [])

        # -------------------------
        # Tables (included table captions and table data)
        # -------------------------

        self.tables = kwargs.get("tables", [])

        # -------------------------
        # Cited references
        # -------------------------

        self.references = kwargs.get("references", [])

    def n_sections(self) -> int:

        """ Returns the number of sections in this article. """

        return len(self.sections)

    def section_title(self, n: int) -> str:

        """ Returns the title of section n or None if n > n_sections-1 """

        if n < len(self.sections):
            return self.sections[n].title
        else: 
            return None

    def n_references(self) -> int:
        """ Number of cited references in the paper. """

        return len(self.references)

    def n_figures(self) -> int:

        """ Number of figures in the paper. """

        return len(self.figures)

    def figure_caption(self, n: int) -> str:

        """ Returns the caption of figure n or None if n > n_figures-1 """

        if n < len(self.figures):
           return self.figures[n].caption
        else:
           return None

    def n_tables(self) -> int:
 
        """ Number of tables in the paper. """

        return len(self.tables)

    def table_caption(self, n: int) -> str:

        """ Returns the caption of table n or None if n > n_tables-1 """

        if n < len(self.tables):
           return self.tables[n].caption
        else:
           return None

    @staticmethod
    def detect_xml_format(tree: etree.Element) -> str:
       
        """ Identify the particular format of xml we are dealing with. """

        xml_format = etree.QName(tree).localname

        if xml_format == "article" or xml_format == "pmc-articleset":

            return "jats"  # paper is in jats PubMed xml

        elif xml_format == "full-text-retrieval-response":

            return "elsevier"  # paper is Elsevier xml

        else:

            return "unknown"

    @classmethod
    def from_xml(cls, tree: etree.Element):

        """ Create an instance of Paper choosing the appropriate parser. """

        fmt = cls.detect_xml_format(tree)

        if fmt == "jats":
            return cls.from_jats_xml(tree)

        if fmt == "elsevier":
            return cls.from_elsevier_xml(tree)

        raise ValueError(f'Unsupported XML format: {fmt}')

    @classmethod
    def from_jats_xml(cls, 
                 tree: etree.Element
                ) -> None:

        """ Create an instance of Paper """

        # -------------------------
        # Check if we have received 
        # the right kind of data
        # -------------------------

        if not isinstance(tree, etree.Element):
            raise PaperException(
            "Received incorrect format: please provide lxml.etree.Element"
            )

        # -------------------------
        # Title
        # -------------------------

        title = tree.xpath(
            "string(//*[local-name()='article-title'])"
        ).strip()

        # -------------------------
        # Article type
        # -------------------------

        article_type = tree.xpath(
            "string(//*[local-name()='article']/@article-type)"
        ).strip()

        # -------------------------
        # DOI
        # -------------------------

        doi = tree.xpath(
            "string(//*[local-name()='article-id'][@pub-id-type='doi'])"
        ).strip()

        # -------------------------
        # PMID
        # -------------------------

        identifiers = {}

        pmid = tree.xpath(
            "string(//*[local-name()='article-id'][@pub-id-type='pmid'])"
        ).strip()

        identifiers["pmid"] = pmid

        # -------------------------
        # PMCID
        # -------------------------

        pmcid = tree.xpath(
            "string(//*[local-name()='article-id'][@pub-id-type='pmc'])"
        ).strip()

        identifiers["pmcid"] = pmcid

        # -------------------------
        # Journal
        # -------------------------

        journal = tree.xpath(
            "string(//*[local-name()='journal-title'])"
        ).strip()

        # -------------------------
        # Publication year
        # -------------------------

        year = tree.xpath(
            "string(//*[local-name()='pub-date']/*[local-name()='year'])"
        ).strip()

        # -------------------------
        # Abstract
        # -------------------------

        abstract = tree.xpath(
            "string(//*[local-name()='abstract'])"
        ).strip()

        # -------------------------
        # Authors
        # -------------------------

        authors = []

        contribs = tree.xpath(
            "//*[local-name()='contrib'][@contrib-type='author']"
        )

        for c in contribs:

            surname = c.xpath(
                "string(.//*[local-name()='surname'])"
            ).strip()

            given = c.xpath(
                "string(.//*[local-name()='given-names'])"
            ).strip()

            full_name = f"{given} {surname}".strip()

            if full_name:
                authors.append(full_name)

        authors = authors

        # -------------------------
        # Sections and paragraphs
        # -------------------------

        sections = []

        sec_nodes = tree.xpath(
            "//*[local-name()='body']/*[local-name()='sec']"
        )

        for sec in sec_nodes:

            sections.append(Section.from_jats_xml(sec))

        # -------------------------
        # Figures (captions only)
        # -------------------------

        figures = []

        figure_nodes = tree.xpath(
            "//*[local-name()='fig']"
        )

        for figure in figure_nodes:

            figures.append(Figure.from_jats_xml(figure))

        # -------------------------
        # Tables (included table captions and table data)
        # -------------------------

        tables = []

        table_nodes = tree.xpath(
            "//*[local-name()='table-wrap']"
        )

        for table in table_nodes:

            tables.append(Table.from_jats_xml(table))

        # -------------------------
        # References 
        # -------------------------

        references = []

        ref_nodes = tree.xpath(
            "//*[local-name()='ref']"
        )

        for ref in ref_nodes:
            references.append(Reference.from_jats_xml(ref))

        # now that we have everything, return instance of Paper

        return cls(        
             title = title,
             doi = doi,
             article_type = article_type,
             identifiers = identifiers,
             journal = journal,
             year = year,
             abstract = abstract,
             sections = sections,
             figures = figures,
             tables = tables,
             references = references
        )

    @classmethod
    def from_elsevier_xml(cls, 
                 tree: etree.Element
                ) -> None:

        """ Create an instance of Paper """

        # -------------------------
        # Check if we have received 
        # the right kind of data
        # -------------------------

        if not isinstance(tree, etree.Element):
            raise PaperException(
            "Received incorrect format: please provide lxml.etree.Element"
            )

        # -------------------------
        # Title
        # -------------------------

        title = tree.xpath(
            "string(//*[local-name()='title'][1])"
        ).strip()

        # -------------------------
        # Article type; for Elsevier type this is empty
        # -------------------------

        article_type = tree.xpath(
            "string(//*[local-name()='article']/@article-type)"
        ).strip()

        # -------------------------
        # DOI
        # -------------------------

        doi = tree.xpath(
            "string(//*[local-name()='doi'])"
        ).strip()

        # -------------------------
        # PII (an Elsevier identifier)
        # -------------------------

        identifiers = {}

        pii = tree.xpath(
            "string(//*[local-name()='pii'])"
        ).strip()

        identifiers["pii"] = pii

        # -------------------------
        # EID (another Elsevier identifier)
        # -------------------------

        eid = tree.xpath(
            "string(//*[local-name()='eid'])"
        ).strip()

        identifiers["eid"] = eid

        # -------------------------
        # Journal
        # -------------------------

        journal = tree.xpath(
            "string(//*[local-name()='publicationName'])"
        ).strip()

        # -------------------------
        # Publication year
        # -------------------------

        year = tree.xpath(
            "string(//*[local-name()='year-nav'])"
        ).strip()

        # -------------------------
        # Abstract
        # -------------------------

        abstract = tree.xpath(
            "string(//*[local-name()='description'])"
        ).strip()

        # -------------------------
        # Authors
        # -------------------------

        author_groups = tree.xpath(
            "//*[local-name()='author-group']"
        )

        authors = []

        for group in author_groups:

            for author in group.xpath("./*[local-name()='author']"):

                given = author.xpath(
                    "string(./*[local-name()='given-name'])"
                ).strip()

                surname = author.xpath(
                    "string(./*[local-name()='surname'])"
                ).strip()

            full_name = f'{given} {surname}'.strip()

            if full_name:
                authors.append(full_name)

        # -------------------------
        # Sections and paragraphs
        # -------------------------

        sections = []

        sec_nodes = tree.xpath(
            "//*[local-name()='section']"
        )

        for sec in sec_nodes:

            sections.append(Section.from_elsevier_xml(sec))

        # -------------------------
        # Figures (captions only)
        # -------------------------

        figures = []

        figure_nodes = tree.xpath(
            "//*[local-name()='figure']"
        )

        for figure in figure_nodes:

            figures.append(Figure.from_elsevier_xml(figure))

        # -------------------------
        # Tables (included table captions and table data)
        # -------------------------

        tables = []

        table_nodes = tree.xpath(
            "//*[local-name()='table']"
        )

        for table in table_nodes:

            tables.append(Table.from_elsevier_xml(table))

        # -------------------------
        # References 
        # -------------------------

        references = []

        ref_nodes = tree.xpath(
            "//*[local-name()='bib-reference']"
        )

        for ref in ref_nodes:
            references.append(Reference.from_elsevier_xml(ref))

        # now that we have everything, return instance of Paper

        return cls(        
             title = title,
             doi = doi,
             article_type = article_type,
             identifiers = identifiers,
             journal = journal,
             year = year,
             abstract = abstract,
             sections = sections,
             figures = figures,
             tables = tables,
             references = references
        )

    def to_dict(self) -> dict:

        """ 
        Return a dictionary representation of the current instance 

        The dictionary representation is a complete representation of the instance; all
        structures must be preserved in the dictionary for subsequent json serialisation.

        """

        return {
            "title": self.title,
            "article_type": self.article_type,
            "doi": self.doi,
            "identifiers": self.identifiers,
            "journal": self.journal,
            "year": self.year,
            "abstract": self.abstract,
            "authors": self.authors,
            "sections": self.sections,
            "figures": [
                 figure.to_dict() 
                 for figure in self.figures
            ],
            "tables": [
                 table.to_dict() 
                 for table in self.tables
            ],
            "references": [
                 reference.to_dict()   
                 for reference in self.references
            ]
        }

    @classmethod
    def from_dict(cls, paperdict):

        """ 
        Return an instance of Paper from a dictionary representation

        The dictionary representation must be a complete representation of the instance.

        """

        return cls(
            title = paperdict["title"],
            doi = paperdict["doi"],
            identifiers = paperdict["identifiers"],
            article_type = paperdict["article_type"],
            journal = paperdict["journal"],
            year = paperdict["year"],
            abstract = paperdict["abstract"],
            authors = paperdict["authors"],
            sections = paperdict["sections"],
            figures = paperdict["figures"],
            tables = paperdict["tables"],
            references = paperdict["references"]
        )

