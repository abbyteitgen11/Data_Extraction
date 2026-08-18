"""
Every constant the pipeline shares. Nothing here does any work.

The one rule worth remembering: property names are spelled with an underscore
("Melting_point"), because they end up as Neo4j labels and Cypher identifiers
cannot contain spaces.
"""
import os
from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent
XML_FILES = ROOT / "xml"
XML = XML_FILES / "SadeghiDESReview.xml"
DATA = ROOT / "data"


def _load_dotenv(path=ROOT / ".env"):
    """Read KEY=value lines from .env into os.environ. Existing vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Outputs. Everything the pipeline writes lands in data/.
TABLE_CSV = DATA / "table2_with_dois.csv"      # wide: one row per DES mixture
LONG_CSV = DATA / "measurements_long.csv"      # long: one row per measurement
REFERENCES_CSV = DATA / "references.csv"
FIGURES_CSV = DATA / "figures.csv"
COMPONENTS_CSV = DATA / "components.csv"
SECTIONS_LLM_CSV = DATA / "sections_llm.csv"
TABLES_UNHANDLED_CSV = DATA / "tables_unhandled.csv"

# Caches. Expensive to rebuild, so they are kept separate from the outputs.
REFERENCE_CACHE = DATA / "reference_map.json"      # existing cache, kept in place
COMPONENT_CACHE = DATA / "components_cache.json"
RAW_LLM_DIR = ROOT / "raw_responses"
# Model responses keyed by content. A full prose run costs 30-45 minutes, so this is
# what makes re-assessing a paper after an alias-file change effectively free.
LLM_CACHE_DIR = RAW_LLM_DIR / "cache"

# Human-curated abbreviation map. Lives next to the code (not in data/) because it
# is edited by hand and worth version-controlling; JSON rather than CSV because
# .gitignore excludes *.csv.
COMPONENT_ALIASES = Path(__file__).resolve().parent / "component_aliases.json"
ALIAS_SUGGESTIONS_CSV = DATA / "alias_suggestions.csv"
ALIAS_CANDIDATES_CSV = DATA / "alias_candidates.csv"
COMPONENT_PROPERTIES_CSV = DATA / "component_properties.csv"

# ---------- papers ----------
# There is deliberately NO default paper DOI. A constant here was previously used as
# a fallback whenever a paper's metadata could not be read, which silently stamped one
# paper's identity onto another's data. Identity now comes from des_pipeline.paper.
PAPERS_DIR = DATA / "papers"
PAPERS_CSV = DATA / "papers.csv"
CROSSREF_CACHE = DATA / "crossref_cache.json"     # shared across papers, keyed by DOI
DUPLICATES_CSV = DATA / "duplicate_measurements.csv"
XML_GLOB = "*.xml"

# ---------- Crossref ----------
MAILTO = os.environ.get("CROSSREF_MAILTO", "abigail.teitgen@csic.es")
USER_AGENT = f"DES-KG/1.0 (mailto:{MAILTO})"
MIN_MATCH_SCORE = 40          # Crossref confidence below this is treated as "no match"

# ---------- the property vocabulary ----------
#
# THE place a property is declared. Add an entry here and it flows into the pydantic
# Literal, the Neo4j node label and HAS_<PROPERTY> relationship, the uniqueness
# constraints, and unit handling. Nothing else needs editing.
#
#   pugview          the PubChem PUG-View heading, or None if PubChem has no such heading
#   plausible_range  physically sensible bounds. A value outside is FLAGGED, not
#                    dropped -- checking the three outliers in the first paper against
#                    the source showed all three were faithful transcriptions of what
#                    the paper printed (Tm 2298 C, viscosity 325,000, conductivity
#                    1548), so a bound violation usually means the SOURCE is wrong or
#                    used a different unit, which is a judgement only a human makes.
#                    Seeded from the observed 1st/99th percentiles with headroom.
#                    Refractive index is the tight, well-behaved one, so a violation
#                    there is a strong signal of an extraction bug rather than a typo.
PROPERTIES = {
    "Melting_point":    {"unit": "C",        "pugview": "Melting Point",
                         "plausible_range": (-150, 400)},
    "Boiling_point":    {"unit": "C",        "pugview": "Boiling Point",
                         "plausible_range": (-100, 600)},
    "Density":          {"unit": "g*cm^-3",  "pugview": "Density",
                         "plausible_range": (0.6, 2.5)},
    "Viscosity":        {"unit": "mPa*s",    "pugview": None,
                         "plausible_range": (0.1, 200000)},
    "Conductivity":     {"unit": "mS*cm^-1", "pugview": None,
                         "plausible_range": (0, 200)},
    "Surface_tension":  {"unit": "mN*m^-1",  "pugview": None,
                         "plausible_range": (10, 100)},
    "Refractive_index": {"unit": "",         "pugview": None,
                         "plausible_range": (1.2, 1.8)},
}
PLAUSIBLE_RANGE = {n: s["plausible_range"] for n, s in PROPERTIES.items()}
PROPERTY_NAMES = tuple(PROPERTIES)
PROPERTY_UNITS = {name: spec["unit"] for name, spec in PROPERTIES.items()}

# PUG-View heading -> our property name. Used by the component route.
PUGVIEW_PROPERTIES = {spec["pugview"]: name for name, spec in PROPERTIES.items()
                      if spec["pugview"]}

# Prose writes units however the authors felt like it. Map the spellings we have
# seen onto the table's, so the graph does not end up with g·cm-3, g⋅cm-3 and
# g*cm^-3 as three different units. An unrecognised unit is kept verbatim rather
# than coerced -- a genuinely different scale should be visible, not hidden.
UNIT_ALIASES = {
    "c": "C", "°c": "C", "˚c": "C", "oc": "C", "celsius": "C", "k": "K",
    "g·cm-3": "g*cm^-3", "g⋅cm-3": "g*cm^-3", "g cm-3": "g*cm^-3",
    "g/cm3": "g*cm^-3", "g/cm^3": "g*cm^-3", "g·cm−3": "g*cm^-3", "g/ml": "g*cm^-3",
    # PubChem's own density spellings. NOTE "g/cu m" is a MILLION times smaller and
    # is deliberately absent -- only cubic-centimetre forms belong here.
    "g/cu cm": "g*cm^-3", "g/cm³": "g*cm^-3", "g/cm cu": "g*cm^-3",
    "gm/cu cm": "g*cm^-3", "g/cc": "g*cm^-3", "kg/l": "g*cm^-3", "kg/dm3": "g*cm^-3",
    "mpa·s": "mPa*s", "mpa⋅s": "mPa*s", "mpa s": "mPa*s", "mpa.s": "mPa*s", "cp": "mPa*s",
    "ms·cm-1": "mS*cm^-1", "ms⋅cm-1": "mS*cm^-1", "ms/cm": "mS*cm^-1",
    "ms cm-1": "mS*cm^-1", "ms·cm−1": "mS*cm^-1",
    "mn·m-1": "mN*m^-1", "mn⋅m-1": "mN*m^-1", "mn/m": "mN*m^-1",
    "mn m-1": "mN*m^-1", "mn·m−1": "mN*m^-1",
}

# Footnote markers used to be hard-coded here, copied from one paper's legend. They
# now come from that paper's own legend via profile_table, so a second paper with a
# different convention needs no code change. This is only the fallback for a table
# whose caption states no measurement temperature at all.
DEFAULT_TEMP = 25

# Review directory: the queue and the spot-check verdicts a human fills in.
REVIEW_DIR = DATA / "review"
REVIEW_QUEUE_CSV = REVIEW_DIR / "queue.csv"

# All the ways the table writes "not reported".
DASH = {"–", "—", "-", "−", ""}

# Separator between papers in the multi-valued Source_* columns. Deliberately not
# ";", which already separates authors *within* one paper.
SOURCE_SEP = "|"

# ---------- prose sections worth sending to the LLM ----------
# These six section titles match the six properties in Table 2.
PROPERTY_SECTIONS = {
    "Melting point": "Melting_point",
    "Density": "Density",
    "Viscosity": "Viscosity",
    "Electrical conductivity": "Conductivity",
    "Surface tension": "Surface_tension",
    "Refractive index": "Refractive_index",
}

# ---------- LLM ----------
LLM_BACKEND = os.environ.get("DES_LLM_BACKEND", "ollama")     # "ollama" | "anthropic"
OLLAMA_MODEL = os.environ.get("DES_OLLAMA_MODEL", "qwen3")
ANTHROPIC_MODEL = os.environ.get("DES_ANTHROPIC_MODEL", "claude-sonnet-4-5")

# qwen3 is a reasoning model and ollama turns thinking on by default. We read only
# message.content and discard message.thinking, so every reasoning token is wasted:
# measured on the Melting point section, thinking produced 22,305 characters of
# discarded reasoning for a 1,865-character answer, and took 678 s instead of 158 s.
OLLAMA_THINK = os.environ.get("DES_OLLAMA_THINK", "0") == "1"

# ollama defaults to a 4096-token context regardless of what the model supports
# (qwen3 handles 40960). Prompt plus answer now runs to ~7000 tokens: the section
# text is up to 1900, the instructions ~1200, and a full answer 3000-4000 because
# every record carries a verbatim source_text quote.
OLLAMA_NUM_CTX = int(os.environ.get("DES_OLLAMA_NUM_CTX", 16384))

# A runaway guard, not a budget. Set it too low and the JSON is cut off mid-object
# and fails to parse entirely -- 4096 truncated four of the six sections once every
# record had to carry all eight fields.
OLLAMA_NUM_PREDICT = int(os.environ.get("DES_OLLAMA_NUM_PREDICT", 8192))

# In an --steps all run the components step sits before the LLM and can take 20+
# minutes, well past ollama's 5-minute default unload.
OLLAMA_KEEP_ALIVE = os.environ.get("DES_OLLAMA_KEEP_ALIVE", "30m")

# qwen3 ships with repeat_penalty = 1, i.e. none. Copying long verbatim quotes at
# temperature 0 then invites degeneration: one run got stuck mid-quote on
# "...It was shown that the viscos" and emitted "0" until it hit num_predict,
# truncating the whole JSON response. A mild penalty breaks those loops without
# noticeably affecting the structured output.
OLLAMA_REPEAT_PENALTY = float(os.environ.get("DES_OLLAMA_REPEAT_PENALTY", 1.1))

# ---------- Neo4j ----------
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")   # required; no default on purpose
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
