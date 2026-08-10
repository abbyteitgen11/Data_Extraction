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
XML = ROOT / "SadeghiDESReview.xml"
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
REFERENCE_CACHE = ROOT / "reference_map.json"      # existing cache, kept in place
COMPONENT_CACHE = DATA / "components_cache.json"
RAW_LLM_DIR = ROOT / "raw_responses"

# ---------- the paper we are extracting from ----------
REVIEW_DOI = "10.1016/j.molliq.2023.121899"

# ---------- Crossref ----------
MAILTO = os.environ.get("CROSSREF_MAILTO", "abigail.teitgen@csic.es")
USER_AGENT = f"DES-KG/1.0 (mailto:{MAILTO})"
MIN_MATCH_SCORE = 40          # Crossref confidence below this is treated as "no match"

# ---------- Table 2 layout ----------
# Each row of Table 2 has 10 <entry> cells:
#   0 HBA | 1 HBD | 2 molar ratio | 3..8 the six properties | 9 references
PROPERTIES = [
    # (entry index, property name, unit)
    (3, "Melting_point", "C"),
    (4, "Density", "g*cm^-3"),
    (5, "Viscosity", "mPa*s"),
    (6, "Conductivity", "mS*cm^-1"),
    (7, "Surface_tension", "mN*m^-1"),
    (8, "Refractive_index", ""),
]
PROPERTY_NAMES = tuple(name for _, name, _ in PROPERTIES)
PROPERTY_UNITS = {name: unit for _, name, unit in PROPERTIES}

# Table 2's footnote: "At a40 C, b20 C, c60 C, d45 C, e30 C, f35 C, g50 C, h55 C".
# A superscript letter on a value means it was measured at that temperature.
TEMP_MAP = {"a": 40, "b": 20, "c": 60, "d": 45, "e": 30, "f": 35, "g": 50, "h": 55}
DEFAULT_TEMP = 25             # everything without a marker is at 25 C

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
# (qwen3 handles 40960). A single section prompt is already ~2400 tokens, which left
# too little room to generate -- the likely cause of the empty Electrical conductivity
# result and the fabricated Density rows.
OLLAMA_NUM_CTX = int(os.environ.get("DES_OLLAMA_NUM_CTX", 8192))

# A runaway guard, not a budget. The largest real answer was 1302 tokens; setting this
# anywhere near 1024 truncates the JSON mid-object and it fails to parse.
OLLAMA_NUM_PREDICT = int(os.environ.get("DES_OLLAMA_NUM_PREDICT", 4096))

# In an --steps all run the components step sits before the LLM and can take 20+
# minutes, well past ollama's 5-minute default unload.
OLLAMA_KEEP_ALIVE = os.environ.get("DES_OLLAMA_KEEP_ALIVE", "30m")

# ---------- Neo4j ----------
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")   # required; no default on purpose
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
