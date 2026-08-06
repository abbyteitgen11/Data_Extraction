
from lxml.etree import _Element
import re

def cell_text(cell):
    return " ".join(cell.xpath(".//text()")).strip()

def build_header_grid(thead: _Element):

    rows = thead.xpath("./*[local-name()='row']")

    n_rows = len(rows)

    # Determine number of columns from first row
    n_cols = sum(
        1
        for _ in rows[0].xpath("./*[local-name()='entry']")
    )

    grid = [
        [None] * n_cols
        for _ in range(n_rows)
    ]

    for row_idx, row in enumerate(rows):

        col_idx = 0

        for cell in row.xpath("./*[local-name()='entry']"):

            # Skip occupied positions
            while (
                col_idx < n_cols
                and grid[row_idx][col_idx] is not None
            ):
                col_idx += 1

            text = cell_text(cell)

            morerows = int(cell.get("morerows", "0"))

            # Fill current cell
            grid[row_idx][col_idx] = text

            # Fill rows spanned by morerows
            for extra_row in range(1, morerows + 1):

                if row_idx + extra_row < n_rows:

                    grid[row_idx + extra_row][col_idx] = "__SPAN__"

            col_idx += 1

    return grid

class Table:
    """
    This class represents a table from an xml file of a paper. It contains 
    an id, a label, a caption, the column titles and rows of data.

    The class will be rarely instantiated by direct call to the constructor (__init__), but 
    rather from factory methods such as @classmethod from_xml or from_dict.

    The method to_dict() creates and returns a dictionary representation. This is useful for
    later serialisation of the instance.
    """

    def __init__(self, 
                 id: int, 
                 label: str,
                 caption: str,
                 columns: list[str],
                 rows: list, 
                 legend: str = None,
                 raw_xml: _Element = None 
        ) -> None:

        """ Table constructor. """

        self.id = id
        self.label = label
        self.caption = caption
        self.columns = columns
        self.rows = rows
        self.legend = legend
        self.raw_xml = raw_xml

    @classmethod
    def from_jats_xml(cls, element: _Element):

        table_id = element.get("id", "")

        label = element.xpath(
            "string(*[local-name()='label'])"
        ).strip()

        caption = element.xpath(
            "string(*[local-name()='caption'])"
        ).strip()

        table_node = element.xpath(
            ".//*[local-name()='table']"
        )[0]

        # get the column heads

        header_cells = table_node.xpath(
            ".//*[local-name()='thead']"
            "//*[local-name()='tr'][1]"
            "/*[local-name()='th' or local-name='td']"
        )

        if header_cells:

           columns = [
               c.xpath("string(.)").strip()
               for c in header_cells
           ]
   
        # now the rows of data

        rows = []

        for tr in table_node.xpath(
            ".//*[local-name()='tbody']//*[local-name()='tr']"
        ):

            row = [
                cell.xpath("string(.)").strip()  
                for cell in tr.xpath(
                    "./*[local-name()='td' or local-name()='th']"
                )
            ]

            rows.append(row)

        # if the first row actually contained the column names...

        if not columns and rows:

           columns = rows[0]
           rows = rows[1:]

        return cls(
            id = table_id,
            label = label,
            caption = caption,
            columns = columns,
            rows = rows,
            raw_xml = element
        )
     
    @classmethod
    def from_elsevier_xml(cls, element: _Element):

        # ---------------------------
        # Helper functions
        # ---------------------------
        def cell_text(cell):
            """ Flatten mixed XML content inside a cell. """
            return " ".join(cell.xpath(".//text()")).strip()

        def split_inline_unit(text):
            """ Split magnitude and units: Density (g/cm3) -> ('Density', 'g/cm3') """

            match = re.search(r"\((.*?)\)", text)

            if match:
                name = re.sub(r"\(.*?\)", "", text).strip()
                unit = match.group(1).strip()
                return name, unit
            return text, None

        # ----------------------------
        # Metadata
        # ----------------------------

        table_id = element.get("id", "")

        label = element.xpath(
            "string(*[local-name()='label'])"
        ).strip()

        caption = element.xpath(
            "string(*[local-name()='caption'])"
        ).strip()

        legend = element.xpath(
            "string(*[local-name()='legend'])"
        ).strip()

        # ---------------------------
        # Locate table structure
        # ---------------------------
        tgroup = element.xpath(".//*[local-name()='tgroup']")

        if not tgroup:
            return cls(
                table_id=table_id,
                label=label,
                caption=caption,
                columns=[],
                rows=[],
                raw_xml=element
            )

        tgroup = tgroup[0]

        # ---------------------------
        # Extract header
        # ---------------------------

        # header_rows = tgroup.xpath(".//*[local-name()='thead']/*[local-name()='row']")

        # columns = []

        thead = tgroup.xpath(
            "./*[local-name()='thead']"
        )[0]

        grid = build_header_grid(thead)

        columns = []

        header_row = grid[0]

        unit_row = grid[1] if len(grid) > 1 else []

        for i, name in enumerate(header_row):

            unit = None

            if i < len(unit_row):

                candidate = unit_row[i]

                if candidate not in (None, "__SPAN__"):

                    unit = candidate

            columns.append({
                "name": name,
                "unit": unit
            })

        """
        if header_rows:

            # extract header matrix

            header_matrix = [
                row.xpath("./*[local-name()='entry']")
                for row in header_rows
            ]

            # first row = names
            name_cells = header_matrix[0]

            # build sparse unit map from remaining rows
            unit_map = {}

            for r in header_matrix[1:]:
                for i, cell in enumerate(r):
                    txt = cell_text(cell)
                    if txt:
                        unit_map[i] = txt

            # build column schema
            for i, cell in enumerate(name_cells):

                raw_name = cell_text(cell)

                # 1. unit from separate unit row
                unit = unit_map.get(i)

                # 2. fallback: inline unit in header
                if unit is None:
                    raw_name, inline_unit = split_inline_unit(raw_name)
                    unit = inline_unit

                columns.append({
                    "name": raw_name,
                    "unit": unit
                })
        """


        # ---------------------------
        # Extract body rows
        # ---------------------------
        rows = []

        body_rows = tgroup.xpath(
             ".//*[local-name()='tbody']/*[local-name()='row']"
        )

        for row in body_rows:

            row_data = [
                cell_text(cell)
                for cell in row.xpath("./*[local-name()='entry']")
            ]

            rows.append(row_data)

        # ---------------------------
        # Return object
        # ---------------------------

        return cls(
            id = table_id,
            label = label,
            caption = caption,
            legend = legend,
            columns = columns,
            rows = rows,
            raw_xml = element
        )

    def to_dict(self) -> dict:
       
        """ Create a dict representation. """

        # We omit the raw_xml representation for serialisation

        return {
            "id": self.id,
            "label": self.label,
            "caption": self.caption,
            "columns": self.columns,
            "rows": self.rows
        }

    @classmethod
    def from_dict(cls, tabledict):
       
        """ Create instance from dict representation. """

        return cls(
            id = tabledict["id"],
            label = tabledict["label"],
            caption = tabledict["caption"],
            columns = tabledict["columns"],
            rows = tabledict["rows"]
        )

