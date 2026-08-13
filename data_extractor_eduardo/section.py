
from lxml.etree import _Element

class Section:
    """
    This class represents a section from an xml file of a paper. It contains 
    an order (the section number in the paper), a title, and a list of paragraphs in the 
    section. Because it might be useful when extracting information from the section, 
    we keep the raw xml encoding, since this potentially contains more information than the 
    plain ascii.

    The class will be rarely instantiated by direct call to the constructor (__init__), but 
    rather from factory methods such as @classmethod from_xml or from_dict.

    The method to_dict() creates and returns a dictionary representation. This is useful for
    later serialisation of the instance.
    """

    def __init__(self, 
                 title: str,
                 paragraphs: list[str],
                 raw_xml: _Element = None 
        ) -> None:

        """ Section constructor. """

        self.title = title
        self.paragraphs = paragraphs
        self.raw_xml = raw_xml

    @classmethod
    def from_jats_xml(cls, element: _Element):

        # -------------------------
        # Section title 
        # -------------------------

        title = element.xpath(
            "string(./*[local-name()='title'])"
        ).strip()

        # -------------------------
        # Section paragraphs 
        # -------------------------

        paragraph_nodes = element.xpath(
            ".//*[local-name()='p']"
        )

        paragraphs = []

        for p in paragraph_nodes:

            text = p.xpath("string()").strip()

            if text:
               paragraphs.append(text)

        return cls(
            title = title,
            paragraphs = paragraphs,
            raw_xml = element
        )
     
    @classmethod
    def from_elsevier_xml(cls, element: _Element):

        # -------------------------
        # Section title 
        # -------------------------

        title = element.xpath(
            "string(.//*[local-name()='section-title'])"
        ).strip()

        # -------------------------
        # Section paragraphs 
        # -------------------------

        paragraph_nodes = element.xpath(
            ".//*[local-name()='para']"
        )

        paragraphs = []

        for p in paragraph_nodes:

            text = "\n".join(
                p.xpath("string()").strip()
                for p in paragraph_nodes
            )

            if text:
               paragraphs.append(text)

        return cls(
            title = title,
            paragraphs = paragraphs,
            raw_xml = element
        )

    def to_dict(self) -> dict:
       
        """ Create a dict representation. """

        # We omit the raw_xml representation for serialisation

        return {
            "title": self.title,
            "paragraphs": self.paragraphs,
        }

    @classmethod
    def from_dict(cls, sectiondict: dict):
       
        """ Create instance from dict representation. """

        return cls(
            title = sectiondict["title"],
            paragraphs = sectiondict["paragraphs"]
        )

