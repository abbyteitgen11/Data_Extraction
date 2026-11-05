"""
How the reviewed gold set is read. The gold files are never edited; every correction the
review needed is applied here, at read time, and listed in the report.

    EXCLUDED_PAPERS     not scored at all (files overwritten by another paper's)
    IGNORED_FILES       one file of a paper not scored (overwritten)
    VERDICT_ALIASES     spellings of a verdict that mean a standard one
    PROPERTY_ALIASES    gold property names outside the vocabulary, mapped onto it

Verdicts, as eval/ANNOTATION.md defines them:
    correct / fix / added   a real item (gold)
    wrong                   not a real item -- a model predicting it makes an error
    blank                   not reviewed -- not scored
"""
import re

EXCLUDED_PAPERS = {
    "Alotaibi_2024": "its gold files were overwritten by Anđelović's (and tables.csv by "
                     "Azad's); models still process it, predictions written for review",
}
IGNORED_FILES = {
    ("Marco-Velasco_2024", "tables"): "tables.csv was overwritten (it is Solcan's)",
}
VERDICT_ALIASES = {"fixed": "fix", "corrected": "fix", "ok": "correct"}
PROPERTY_ALIASES = {
    # A DES's freezing point is its solid-liquid transition, which the pipeline records
    # as Melting_point (Jablonský Table 3 reports "freezing point").
    "Freezing_point": "Melting_point",
}
REAL = {"correct", "fix", "added"}

NOTES = [
    "Alotaibi_2024 excluded from scoring (gold overwritten).",
    "Marco-Velasco_2024 tables.csv ignored (overwritten).",
    "'fixed' read as 'fix' (Sadeghi tables, Solcan dictionary).",
    "Freezing_point scored as Melting_point (Jablonský, 27 rows).",
    "In tables.csv, 'wrong' means the prediction was wrong; the corrected gold values "
    "are used (Verma Tables 1-3).",
    "Component lists written with ':', '/', '-' or '+ x%' are parsed with the "
    "pipeline's own composition parser.",
    "tables.csv: a table counts as predicted relevant when something was extracted "
    "from it (ledger status 'extracted'), matching 'should be extracted'.",
    "Literature values reprinted in a research paper's table (Lit column or "
    "'(ref. N)') are kept in the data; not in the gold, so unscored extras.",
    "columns.csv: only columns with a known label (correct/fix) are scored; 'wrong' "
    "marks columns of not-relevant tables (and Sadeghi Table 2 col 2, left unlabelled).",
]


def verdict(row, task=""):
    """The standard verdict of a gold row: correct | fix | added | wrong | ''."""
    v = str(row.get("verdict") or "").strip().lower()
    v = VERDICT_ALIASES.get(v, v)
    if v not in ("correct", "fix", "added", "wrong"):
        return ""
    if task == "tables" and v == "wrong":
        return "fix"          # every table is real; the corrected values are the gold
    return v


def components(text):
    """A gold component field -> list of names. ';' is the convention; others parsed."""
    text = str(text or "").strip()
    if not text:
        return [], ""
    if ";" in text:
        return [c.strip() for c in text.split(";") if c.strip()], ""
    from des_pipeline.composition import parse_mixture

    comp = parse_mixture(text)
    names = [c for c in comp.components if c]
    if len(names) < 2 and "-" in text:
        # "Tetrabutylammonium chloride-dodecanol": a hyphen between two words of 5+
        # letters joins two components (Huang). Not used for "1,2-propanediol".
        parts = re.split(r"(?<=[A-Za-z]{5})-(?=[A-Za-z]{5})", text)
        if len(parts) >= 2:
            names = [p.strip() for p in parts]
    return names, comp.ratio_raw


def property_name(name):
    return PROPERTY_ALIASES.get(str(name or "").strip(), str(name or "").strip())
