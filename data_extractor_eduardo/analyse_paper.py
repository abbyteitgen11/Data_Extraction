
from chemdataextractor import Document

from paper import Paper

def analyse_paper(paper: Paper) -> None:

    """ Extracts info from paper using CDE v2 """

    titledoc = Document(paper.abstract)

    # Are any chemicals mentioned in the title?

    titlechem = [c.text for c in titledoc.cems]

    # any records?

    titlerecords = [r.serialize() for r in titledoc.records]

    print(paper.abstract)

    print(f'Chemicals in title: {titlechem}')
    print(f'Records in title: {titlerecords}')

    print()
