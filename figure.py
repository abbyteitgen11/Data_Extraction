
from lxml import etree

class Figure:
    """
    This class represents a figure from an xml file of a paper. It does not contain graphical
    information, only an id, a label and a caption.

    The class will be rarely instantiated by direct call to the constructor (__init__), but 
    rather from factory methods such as @classmethod from_xml or from_dict.

    The method to_dict() creates and returns a dictionary representation. This is useful for
    later serialisation of the instance.
    """

    def __init__(self, 
                 id: int, 
                 label: str,
                 caption: str
        ) -> None:

        """ Figure constructor. """

        self.id = id
        self.label = label
        self.caption = caption

    def to_dict(self) -> dict:

        """ Create a dict representation. """

        return {
           "id": self.id,
           "label": self.label,
           "caption": self.caption
        }

    @classmethod
    def from_jats_xml(cls, element):

        """ Create an instance of Figure from PubMed (JATS) xml data. """

        fig_id = element.get("id", "")

        label = element.xpath(
            "string(*[local-name()='label'])"
        ).strip()

        caption = element.xpath(
            "string(*[local-name()='caption'])"
        ).strip()

        return cls(
            id = fig_id,
            label = label,
            caption = caption
        )

    @classmethod
    def from_elsevier_xml(cls, element):

        """ Create an instance of Figure from Elsevier xml data. """

        fig_id = element.get("id", "")

        label = element.xpath(
            "string(*[local-name()='label'])"
        ).strip()

        caption = element.xpath(
            "string(*[local-name()='caption'])"
        ).strip()

        return cls(
            id = fig_id,
            label = label,
            caption = caption
        )

    @classmethod
    def from_dict(cls, figdict: dict):

        """ Create an instance of Figure from a dictionary. """

        return cls(
            id = figdict["id"],
            label = figdict["label"],
            caption = figdict["caption"]
        )
