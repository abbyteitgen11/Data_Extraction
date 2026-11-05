"""
The paper's own definitions, gathered once, before anything is extracted.

A table that names its mixtures "CE", "CEM1", "CEM5", a figure that says "DES5", a
results table whose rows read "DES 1:2", "DES 1:4 + 10 %wt EC" -- each of these means
exactly what the paper says it means, and the meaning is usually somewhere else in the
paper: in a definitions table, a glossary, the methods section, or a phrase like
"choline chloride (ChCl)". Every route used to resolve names on its own, against a
corpus-wide vocabulary, and none of them could see those definitions, so a code was
either dropped or kept as an unexplained "component".

This module builds one dictionary per paper, from four sources, most authoritative
first (an earlier source is never overwritten by a later one):

    definitions_table   a table whose job is to say what each DES is (Fan's Table 2)
    glossary            the paper's abbreviation list (Marco-Velasco's back matter)
    text                "Full name (ABBR)" pairs, by the Schwartz-Hearst algorithm
    model               one call over title, abstract and methods: which DES does this
                        paper prepare or study, and what did it use them for?

Every model entry carries its evidence quote and is accepted only if that quote and
every component name it gives are actually in the paper. The result is written to
data/papers/<slug>/des_dictionary.csv, where a human can read and correct it.

A paper that studies ONE family of DES and names it only "DES" (Dinh: LiTFSI:TFA at
1:2..1:5) gets a `family`, so "DES 1:4" resolves to LiTFSI:TFA (1:4).
"""
import copy
import re

from pydantic import ValidationError

from . import config
from .composition import Composition, compose, is_water, parse_mixture
from .schema import DictionaryRow, PaperOverview
from .textmatch import normalise, quote_found

OVERVIEW_PROMPT = """You read the opening of a scientific paper and report which deep \
eutectic solvents (DES) it is about, and what it used them for. You never guess: \
everything you return must be stated in the text below.

{text}

Return:

des_is_subject   true when DES are a main subject of this paper -- it prepares,
                 measures or uses them, or it is a review whose topic is DES. false
                 when DES are mentioned only in passing.

des              every DES this paper itself prepares, measures, uses or tabulates.
                 For a review, the ones it discusses by name. At most 40.
                 For each:
                   code         the paper's own name or code for it ("DES1", "N3",
                                "CEM5"), exactly as written. null if it has none.
                   components   its constituents, hydrogen-bond acceptor first,
                                EXACTLY as the paper writes them, abbreviations
                                included. Never a drug, solute, analyte, substrate,
                                biomass or metal that the DES acts on -- those are what
                                the DES is used FOR. Water is a component only when the
                                DES is made with it.
                   molar_ratio  as written ("1:2"). null if not stated. When the paper
                                studies one DES at several ratios, return one record per
                                ratio.
                   water_percent  the water content the DES was made with, in percent,
                                if stated.
                   evidence     the words that define it, copied verbatim, under 200
                                characters.
                 An ionic liquid used on its own, without a hydrogen-bond donor, is not a
                 DES: leave it out.

applications     each use the paper reports for its DES. For each:
                   use       in the paper's own words: what the DES was used for.
                   des       which DES from your list were USED for it -- their code,
                             or their components joined with ":". Not DES the paper
                             only cites or compares against.
                   role      solvent, extractant, electrolyte, catalyst, sorbent, ...
                   target    what the DES acted on: the drug, analyte, feedstock or
                             material.
                   evidence  the words stating the use, copied verbatim, under 200
                             characters.

If the text names no DES, return empty lists.
"""

_METHODS = re.compile(r"material|method|experiment|preparation|synthes|chemical|reagent|"
                      r"procedure", re.I)
_GENERIC = re.compile(r"^\s*(?:the\s+)?(?:pure\s+)?(?:na)?des(?:s)?\s*$|"
                      r"^\s*deep\s+eutectic\s+solvents?\s*$", re.I)


_DES_WORDS = re.compile(r"\bdes(s)?\b|eutectic|\bnades\b|\bthedes\b", re.I)
_IL_CAPTION = re.compile(r"\bils?\b|ionic liquid", re.I)


def mentions_des(table, paragraphs=None):
    """Does the table -- caption, legend, header, or what the paper says about it --
    say it is about DES? A table that never does (corrosion inhibitors, hemicellulose
    sugars, ionic-liquid processes) is accepted only through rows the paper defines."""
    from .profile_table import table_context

    header = " ".join(" ".join(table.column(i)) for i in range(table.n_columns))
    text = f"{table.caption} {table.footnotes} {header} " \
           f"{table_context(table, paragraphs or [])}"
    return bool(_DES_WORDS.search(text))


def caption_mixes_il_and_des(table):
    """'ILs/DESs Catalyzed Lignin Depolymerization': both, with no row saying which."""
    caption = f"{table.caption} {table.footnotes}"
    return bool(_IL_CAPTION.search(caption)) and bool(_DES_WORDS.search(caption))


_LEGEND_ITEM = re.compile(r"^\s*(\[?[A-Za-z0-9\[\]()+,.'-]{1,16}?)\s*[–—=:]\s*"
                          r"([A-Za-z0-9][^–—=:]{2,80})$")


def legend_definitions(legend):
    """'U–Urea, Gly–Glycerol, ChCl–Choline chloride' -> [(abbr, full name)].

    A table legend that lists abbreviations, separated by an en/em dash, '=' or ':'
    (never a plain hyphen, which is inside chemical names). Items are split on ', ' or
    '; ' only where the next item again has the 'ABBR–' shape, because full names can
    contain commas ("2,2,2-Trifluoroacetamide").
    """
    text = str(legend or "")
    if not re.search(r"[–—=]", text):
        return []
    # A caption lists them in parentheses: "... at 298 K (AA = acetamide; U = urea)".
    text = re.sub(r"(?:(?<=\s)|^)\(\s*(?=\[?[A-Za-z0-9\[\]()+,.'-]{1,16}?\s*[–—=])", "; ", text)
    pieces = re.split(r"[,;]\s+(?=\[?[A-Za-z0-9\[\]()+,.'-]{1,16}?\s*[–—=:])", text)
    out = []
    for piece in pieces:
        match = _LEGEND_ITEM.match(piece.strip().rstrip("."))
        if not match:
            continue
        abbr, full = match.group(1).strip(), match.group(2).strip()
        if full.count(")") > full.count("("):
            full = full[:full.rindex(")")].strip()
        if len(abbr) < len(full) and re.search(r"[a-z]{3}", full) \
                and not re.fullmatch(r"\d+(\.\d+)?", abbr):
            out.append((abbr, full))
    return out


def code_key(text):
    """'[ChCl][Gl]3' -> 'chclgl3', 'DES 1' -> 'des1', 'CEM5' -> 'cem5'."""
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


class PaperDictionary:
    """Codes and abbreviations one paper defines. Lookups never touch the network."""

    def __init__(self, paper_key=""):
        self.paper_key = paper_key
        self.mixtures = {}          # code_key -> Composition (with .code and .source)
        self.aliases = {}           # abbreviation (lower) -> full name
        self.family = None          # Composition with no ratio, when there is one family
        self.des_is_subject = None
        self.rows = []              # DictionaryRow, for the CSV
        self.table_counts = {}      # table id -> definitions it contributed
        self.overview_text = ""     # what the overview call read, for verifying quotes

    # --- building ---
    def identity(self, comp):
        """(component set, ratio) -- what makes two entries the same DES."""
        names = frozenset(self.name_key(c) for c in comp.components
                          if c and not is_water(c))
        ratio = re.sub(r"\s+", "", comp.ratio_molar or comp.ratio_raw or "")
        # Water and additives are part of what the DES is: Solcan's N12 and N13 share
        # components and HBA:HBD ratio and differ only in water content.
        water = (round(comp.water_fraction, 4) if comp.water_fraction is not None
                 else "present" if any(is_water(c) for c in comp.components) else None)
        additives = tuple(sorted((k.lower(), round(v[0], 4)) for k, v in comp.additives.items()))
        return names, ratio, water, additives

    def add_mixture(self, code, comp, source, evidence="", verified=True, note=""):
        comp = copy.deepcopy(comp)
        comp.code, comp.source = (code or "").strip(), source
        if verified:
            # The same DES written two ways ("ChCl/U" from a table, "choline chloride:
            # urea" from the overview) is one entry: the new code points at the first.
            ident = self.identity(comp)
            if len(ident[0]) >= 2:
                for existing in self.mixtures.values():
                    if self.identity(existing) == ident and existing.code != comp.code:
                        key = code_key(comp.code)
                        if key:
                            self.mixtures.setdefault(key, existing)
                        existing.note(f"also written as {comp.code!r} ({source})")
                        # The code keeps its own row, so every name the paper uses for
                        # the DES is listed -- but it resolves to the one entry.
                        self.rows.append(DictionaryRow(
                            Paper_key=self.paper_key, kind="mixture", code=comp.code,
                            components=";".join(existing.components),
                            components_written=";".join(comp.components_written),
                            ratio_raw=existing.ratio_raw, ratio_molar=existing.ratio_molar,
                            water_fraction=existing.water_fraction,
                            water_basis=existing.water_basis, source=source,
                            evidence=str(evidence or "")[:300], verified=True,
                            note=f"same DES as {existing.code!r}"))
                        return existing
        self.rows.append(DictionaryRow(
            Paper_key=self.paper_key, kind="mixture", code=comp.code,
            components=";".join(comp.components),
            components_written=";".join(comp.components_written),
            ratio_raw=comp.ratio_raw, ratio_molar=comp.ratio_molar,
            water_fraction=comp.water_fraction, water_basis=comp.water_basis,
            source=source, evidence=str(evidence or "")[:300], verified=verified,
            note="; ".join([note] + comp.notes if note else comp.notes)[:300]))
        key = code_key(comp.code)
        if verified and key and key not in self.mixtures:
            self.mixtures[key] = comp
        return comp

    def add_alias(self, abbr, full, source, evidence=""):
        abbr, full = str(abbr or "").strip(), str(full or "").strip()
        if not abbr or not full or abbr.lower() == full.lower():
            return
        self.rows.append(DictionaryRow(
            Paper_key=self.paper_key, kind="alias", code=abbr, components=full,
            source=source, evidence=str(evidence or "")[:300]))
        self.aliases.setdefault(abbr.lower(), full)

    # --- using ---
    def lookup(self, text):
        """A code or generic name -> a copy of its Composition, or None."""
        key = code_key(text)
        if key and key in self.mixtures:
            return copy.deepcopy(self.mixtures[key])
        inferred = self._numbered_variant(key)
        if inferred is not None:
            return inferred
        if self.family is not None and _GENERIC.match(str(text or "")):
            out = copy.deepcopy(self.family)
            out.code = "DES"
            return out
        return None

    def _numbered_variant(self, key):
        """'cem5' when the paper defines 'CEM' as 1:4:(1-11) -> that DES at 1:4:5.

        Alotaibi's Table 1 defines CEM as ChCl:EG:malonic acid "1:4:(1-11)" and its
        Table 2 then reports CEM1, CEM5, CEM10: the suffix is the varied ratio part.
        Inferred only when the definition really leaves its last part open (a range or
        a gap) and the suffix falls inside the stated range; recorded as `inferred`.
        """
        match = re.fullmatch(r"([a-z]+)(\d+(?:\.\d+)?)", key or "")
        if not match or match.group(1) not in self.mixtures:
            return None
        base = self.mixtures[match.group(1)]
        if not base.parts or base.parts[-1] is not None:
            return None
        value = float(match.group(2))
        span = re.search(r"\(?(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)\)?\s*$",
                         base.ratio_raw or "")
        if span and not float(span.group(1)) <= value <= float(span.group(2)):
            return None
        out = copy.deepcopy(base)
        fixed = [p for p in base.parts[:-1]]
        out.parts = fixed + [value]
        out.ratio_raw = ":".join(f"{p:g}" for p in out.parts)
        out.ratio_molar = out.ratio_raw if len(out.parts) == len(
            [c for c in out.components if not is_water(c)]) else ""
        out.ratio_source = "inferred"
        out.code = f"{base.code}{match.group(2)}"
        out.note(f"inferred from the code's suffix and {base.code}'s ratio "
                 f"{base.ratio_raw!r}")
        return out

    def name_key(self, name):
        """A comparable key for one component name: abbreviation expanded, alnum only."""
        text = str(name or "").strip()
        full = self.resolve(text) or text
        if _GLOBAL_ALIASES:
            full = _GLOBAL_ALIASES.get(full.lower(), full)
        return re.sub(r"[^a-z0-9]", "", full.lower())

    def matches(self, comp):
        """Is this composition one of the DES the paper defines? -> bool.

        By code, or by the same set of non-water constituents as a defined entry
        (ratio not required -- the definition and the data table often differ in how
        they state it).
        """
        if comp.code and (code_key(comp.code) in self.mixtures
                          or self.lookup(comp.code) is not None):
            return True
        names = frozenset(self.name_key(c) for c in comp.components
                          if c and not is_water(c))
        if len(names) < 2:
            return False
        for entry in list(self.mixtures.values()) + ([self.family] if self.family else []):
            other = frozenset(self.name_key(c) for c in entry.components
                              if c and not is_water(c))
            if other and (other == names or (entry is self.family and other <= names)):
                return True
        return False

    def best_match(self, comp):
        """The dictionary DES this composition is. -> (entry, ratio_agrees) or (None, False)."""
        if comp.code:
            entry = self.lookup(comp.code)
            if entry is not None:
                return entry, True
        ident = self.identity(comp)
        names, ratio = ident[0], ident[1]
        if len(names) < 2:
            return None, False
        candidates = [e for e in self.mixtures.values() if self.identity(e)[0] == names]
        if not candidates:
            return None, False
        # Same water and additives first: N12 and N13 differ only in water content.
        candidates.sort(key=lambda e: self.identity(e)[2:] != ident[2:])
        for entry in candidates:
            if ratio and self.identity(entry)[1] == ratio:
                return entry, True
        no_ratio = [e for e in candidates if not self.identity(e)[1]]
        return (no_ratio or candidates)[0], not ratio

    def canonicalize(self, comp):
        """Give a row's DES the names (and, when the ratio agrees, the code) the paper
        itself defines. In place; returns comp.

        A row read as "choline chloride:U" and the dictionary entry "Choline chloride:
        Urea" (code ChCl/U) are one DES, and the data should say so: same names, same
        code, so it joins the rest of the paper's data on that DES.
        """
        entry, ratio_agrees = self.best_match(comp)
        if entry is None:
            return comp
        by_key = {self.name_key(c): c for c in entry.components if c}
        comp.components = [by_key.get(self.name_key(c), c) if not is_water(c) else c
                           for c in comp.components]
        if entry.code and (ratio_agrees or not comp.ratio_raw) and not comp.code:
            comp.code = entry.code
        if not comp.ratio_raw and entry.ratio_raw:
            comp.ratio_raw, comp.parts = entry.ratio_raw, list(entry.parts)
            comp.ratio_molar = entry.ratio_molar
            comp.ratio_source = comp.ratio_source or "dictionary"
        return comp

    @property
    def has_definitions(self):
        """Does the paper define at least one real DES (not just abbreviations)?"""
        return any(len([c for c in m.components if not is_water(c)]) >= 2
                   for m in self.mixtures.values())

    def resolve(self, name):
        """An abbreviation this paper defines -> its full name, or None.

        Matched ignoring hyphens, spaces and dots too: a legend defines "MeU" and the
        table cells write "Me-U".
        """
        key = str(name or "").strip().lower()
        if key in self.aliases:
            return self.aliases[key]
        squeezed = re.sub(r"[\s.\-–\[\]]", "", key)
        if not squeezed:
            return None
        for abbr, full in self.aliases.items():
            if re.sub(r"[\s.\-–\[\]]", "", abbr) == squeezed:
                return full
        # A stereo prefix on a defined abbreviation: "D-Glu" with "Glu–Glucose".
        stereo = re.fullmatch(r"((?:d|l|dl|r|s|\(±\)|\([rs+-]\))-)(.+)", key)
        if stereo and stereo.group(2) in self.aliases:
            return stereo.group(1).upper() + self.aliases[stereo.group(2)]
        return None

    def resolver(self, base=None):
        """A name resolver: this paper's own aliases first, then `base` (corpus-wide)."""
        def resolve(name):
            found = self.resolve(name)
            if found:
                return (base(found) if base else None) or found
            return base(name) if base else None
        return resolve

    def to_rows(self):
        return list(self.rows)


_GLOBAL_ALIASES = None


def global_alias_resolver():
    """The hand-curated alias file (component_aliases.json) as a resolver.

    Deterministic and offline: an abbreviation resolves only if a human put it in that
    file. Used by the table and definition routes, where a name must never change
    depending on which other papers happen to have been processed first.
    """
    global _GLOBAL_ALIASES
    if _GLOBAL_ALIASES is None:
        from .enrich_components import load_aliases

        try:
            _GLOBAL_ALIASES = load_aliases()
        except Exception:
            _GLOBAL_ALIASES = {}

    def resolve(name):
        return _GLOBAL_ALIASES.get(str(name or "").strip().lower())
    return resolve


def corpus_resolver(index):
    """The corpus-wide resolver: global alias file + spellings already in the corpus.

    Never consults PubChem: resolution inside extraction must be deterministic and
    offline, and an unknown name is kept as written rather than guessed at.
    """
    if index is None:
        return None
    from .enrich_components import resolve_component

    def resolve(name):
        found, _how = resolve_component(name, index, allow_lookup=False)
        return found
    return resolve


# ---------- building ----------
def overview_text(paper, root, dialect, sections, tables, max_methods=5000):
    """What the overview call reads: title, keywords, abstract, methods, captions."""
    parts = [f"TITLE: {paper.title}"]
    keywords = dialect.keywords(root) if hasattr(dialect, "keywords") else []
    if keywords:
        parts.append("KEYWORDS: " + "; ".join(keywords))
    abstract = dialect.abstract(root) if hasattr(dialect, "abstract") else ""
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
    methods, used = [], 0
    for _sid, title, text in sections:
        if _METHODS.search(title or "") and used < max_methods:
            chunk = re.sub(r"\s+", " ", text)[:max_methods - used]
            methods.append(f"[{title}] {chunk}")
            used += len(chunk)
    if not methods:                      # no methods section: the opening of the body
        for _sid, title, text in sections[:3]:
            if used >= max_methods // 2:
                break
            chunk = re.sub(r"\s+", " ", text)[:max_methods // 2 - used]
            methods.append(f"[{title}] {chunk}")
            used += len(chunk)
    if methods:
        parts.append("METHODS EXCERPT: " + " ".join(methods))
    captions = [f"{t.label}: {t.caption}" for t in tables if t.caption][:20]
    if captions:
        parts.append("TABLE CAPTIONS: " + " | ".join(c[:200] for c in captions))
    return "\n\n".join(parts)


def paper_overview(paper, text, backend=None, refresh=False):
    """One model call per paper. -> PaperOverview, or None if the answer is unusable."""
    from .extract_text_llm import cached_call_llm

    prompt = OVERVIEW_PROMPT.format(text=text)
    section_id = re.sub(r"[^A-Za-z0-9_.-]", "-", f"paper-{paper.slug}")[:48]
    raw, _cached = cached_call_llm(prompt, PaperOverview.model_json_schema(),
                                   section_id=section_id, backend=backend,
                                   refresh=refresh)
    try:
        return PaperOverview.model_validate_json(raw)
    except ValidationError as exc:
        print(f"    overview did not validate: {exc.errors()[0]['msg']}")
        return None


def _named_in(name, haystack_norm, dictionary):
    """Is this component name actually written in the paper (or defined by it)?"""
    n = normalise(name)
    if not n:
        return False
    if is_water(name) or n in haystack_norm or dictionary.resolve(name):
        return True
    # "Choline chloride" written as "choline-chloride" or with a stray space
    squeezed = re.sub(r"[\s\-]", "", n)
    return bool(squeezed) and squeezed in re.sub(r"[\s\-]", "", haystack_norm)


def build(paper, root, dialect, sections, tables, profiles, index=None, backend=None,
          refresh=False, use_model=True):
    """-> (PaperDictionary, PaperOverview | None)."""
    from .extract_table import extract_definitions
    from .suggest_aliases import harvest_definitions

    d = PaperDictionary(paper.key)

    # 1. the paper's glossary
    for abbr, full in dialect.glossary(root):
        comp = parse_mixture(full)
        if comp.n_non_water >= 2:
            d.add_mixture(abbr, comp, "glossary", f"{abbr}: {full}")
        else:
            d.add_alias(abbr, full, "glossary", f"{abbr}: {full}")

    # 1b. abbreviation lists in table legends: "U–Urea, Gly–Glycerol, ChCl–Choline
    #     chloride" (Jablonský Table 2 defines every abbreviation its rows use this way)
    #     or in the caption: "(AA = acetamide; MA = malic acid)" (Jablonský Table 3)
    for table in tables:
        for abbr, full in (legend_definitions(table.footnotes)
                           + legend_definitions(table.caption)):
            if abbr.lower() not in d.aliases:
                d.add_alias(abbr, full, "legend", f"{table.label}: {abbr}–{full}")

    # 2. "Full name (ABBR)" written in the running text and the abstract
    abstract = dialect.abstract(root) if hasattr(dialect, "abstract") else ""
    for found in harvest_definitions(list(sections) + [("abstract", "", abstract)],
                                     paper.key):
        if found["status"] == "candidate" and found["abbreviation"].lower() not in d.aliases:
            d.add_alias(found["abbreviation"], found["name"], "text", found["evidence"])

    # 2b. single-letter abbreviations ("urea (U)", "glycerol (G)")
    for abbr, full in single_letter_aliases([abstract] + [t for _s, _ti, t in sections]):
        if abbr.lower() not in d.aliases:
            d.add_alias(abbr, full, "text", f"{full} ({abbr})")

    resolve = d.resolver(global_alias_resolver())

    # 3. tables whose job is to define the DES
    paragraphs = dialect.paragraphs(root) if hasattr(dialect, "paragraphs") else []
    for table in tables:
        profile = profiles.get(table.id)
        if profile is None or not profile.relevant \
                or "des_definitions" not in profile.content_types:
            continue
        if not mentions_des(table, paragraphs) or caption_mixes_il_and_des(table):
            continue                    # a reagents / inhibitors / sugars table
        found = extract_definitions(table, profile, paper, resolve=resolve,
                                    dictionary=None)
        d.table_counts[table.id] = len(found)
        for code, comp in found.items():
            d.add_mixture(code, comp, "definitions_table",
                          f"{table.label}: {comp.written}"[:300])
            for abbr, full in comp.aliases.items():
                if abbr.lower() not in d.aliases:
                    d.add_alias(abbr, full, "definitions_table",
                                f"{table.label}: {full} ({abbr})")

    # 4. one model call: which DES, and what for
    overview = None
    if use_model:
        text = overview_text(paper, root, dialect, sections, tables)
        d.overview_text = text
        overview = paper_overview(paper, text, backend=backend, refresh=refresh)
    if overview is not None:
        d.des_is_subject = overview.des_is_subject
        haystack = normalise(text)
        for draft in overview.des:
            missing = [c for c in draft.components if not _named_in(c, haystack, d)]
            evidence_ok = quote_found(draft.evidence, text)
            comp = compose(draft.components, draft.molar_ratio or "", "Molar ratio",
                           water_text=(f"{draft.water_percent}"
                                       if draft.water_percent is not None else ""),
                           water_header="water %", resolve=resolve)
            verified = evidence_ok and not missing and comp.n_non_water >= 2
            note = ("; ".join(
                (["evidence not found in the paper"] if not evidence_ok else [])
                + ([f"not in the paper: {', '.join(missing)}"] if missing else [])
                + (["fewer than two non-water components"]
                   if comp.n_non_water < 2 else [])))
            code = draft.code or comp.label()
            d.add_mixture(code, comp, "model", draft.evidence, verified=verified,
                          note=note)

    _keep_des_aliases(d, tables, profiles, sections, abstract)
    _set_family(d)
    return d, overview


_SINGLE_LETTER = re.compile(r"\b((?:[A-Za-z][\w\-,]*\s+){0,2}[A-Za-z][\w\-,]*)\s*\(\s*([A-Z])\s*\)")
_CHEMICAL_WORD = re.compile(
    r"(acid|chloride|bromide|iodide|fluoride|nitrate|sulfate|sulphate|phosphate|acetate|"
    r"glycol|glycerol|urea|amine|amide|ammonium|phosphonium|imidazolium|choline|betaine|"
    r"anol|diol|triol|ol|ine|ate|ide|ose|one)$", re.I)


def chemical_name(text):
    """Does this read like a chemical name? PubChem cache first, then its ending."""
    from .composition import _cid_names

    t = str(text or "").strip().lower()
    if not t or len(t.split()) > 5:
        return False
    if t in _cid_names():
        return True
    last = re.split(r"[\s-]+", t)[-1]
    return len(last) >= 4 and bool(_CHEMICAL_WORD.search(last))


def single_letter_aliases(texts):
    """'urea (U)', 'glycerol (G)' -> [(abbr, name)]. Schwartz-Hearst needs 2+ letters.

    The letter must be the initial of the name's first word, and the name must read
    as a chemical, so enumerations like "(a)" and "(B)" never qualify.
    """
    out = []
    for text in texts:
        for match in _SINGLE_LETTER.finditer(str(text or "")):
            words, letter = match.group(1).split(), match.group(2)
            for start in range(len(words)):
                name = " ".join(words[start:]).strip(" ,")
                if name[:1].lower() == letter.lower() and chemical_name(name):
                    out.append((letter, name))
                    break
    return out


def _keep_des_aliases(d, tables, profiles, sections, abstract):
    """Keep only abbreviations the paper uses for a DES constituent. In place.

    The abbreviation harvest finds every "Full name (ABBR)" in the paper, and most are
    not chemistry at all -- PWF (pure water flux), SLM, EMVs, SIBs -- or are chemistry
    unrelated to any DES (CO2, CH4). They are harmless to extraction but fill
    des_dictionary.csv. An abbreviation is kept when it appears in a DES context: a
    component or mixture cell of a table, a DES formula in the text ("ChCl:U", "U/Gly"),
    or the code or written components of a DES the paper defines. Resolution still
    sees every candidate; only the file is filtered.
    """
    context = []
    for table in tables:
        profile = profiles.get(table.id)
        if profile is None:
            continue
        cols = [c.index for c in profile.columns if c.role in ("component", "mixture")]
        context += [row[i].text for row in table.rows for i in cols if i < len(row)]
    for comp in d.mixtures.values():
        context += [comp.code or "", comp.written or ""] + list(comp.components_written)
    formulas = [m.group(0) for _sid, _t, text in sections
                for m in re.finditer(r"[\w\[\]()+-]+\s*[:/]\s*[\w\[\]()+-]+(?:\s*[:/]\s*"
                                     r"[\w\[\]()+-]+)*", str(text or ""))]
    haystack = " | ".join(context + formulas)
    squeezed = re.sub(r"[\s.\-–]", "", haystack.lower())
    kept = []
    for row in d.rows:
        if row.kind != "alias" or row.source == "definitions_table":
            kept.append(row)
            continue
        abbr = re.escape(row.code)
        if re.search(rf"(?<![\w]){abbr}(?![\w])", haystack) or (
                len(row.code) >= 3
                and re.sub(r"[\s.\-–]", "", row.code.lower()) in squeezed):
            kept.append(row)
    dropped = len(d.rows) - len(kept)
    if dropped:
        print(f"    {dropped} abbreviation(s) unrelated to any DES left out of the dictionary")
    d.rows = kept


def _is_generic(name):
    return bool(_GENERIC.match(str(name or "")))


def _set_family(d):
    """One family studied at several ratios -> a bare "DES" means that family.

    Entries that themselves say only "DES" (Dinh's "DES 1:4 + 10 %wt EC") are ignored
    when finding the family, then rewritten with its components.
    """
    from collections import Counter

    # The family is the component set most of the paper's DES-coded entries share
    # ("DES1:4", "DES 1:2" -> LiTFSI:TFA). A single other entry -- Dinh's conventional
    # EC-DMC electrolyte -- no longer prevents finding it.
    entries = [c for c in {id(v): v for v in d.mixtures.values()}.values()
               if not any(_is_generic(x) for x in c.components)]
    coded = [c for c in entries if re.search(r"\bDES", c.code or "", re.I)] or entries
    counts, first = Counter(), {}
    for comp in coded:
        names = frozenset(normalise(c) for c in comp.components if not is_water(c))
        if len(names) >= 2:
            counts[names] += 1
            first.setdefault(names, comp)
    if not counts:
        return
    ranked = counts.most_common(2)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return                                # a tie: no single family
    if len(counts) > 1 and ranked[0][1] < 2:
        return                                # several sets, none repeated
    only = first[ranked[0][0]]
    base = [c for c in only.components if not is_water(c)]
    d.family = Composition(written="DES", components=base,
                           components_written=list(base), source="family")
    for comp in d.mixtures.values():
        if any(_is_generic(c) for c in comp.components):
            rest = [c for c in comp.components if not _is_generic(c)]
            comp.components = list(base) + rest
            if comp.parts and len(comp.parts) == len(rest) + 1:
                comp.parts = []                # the ratio named the family, not "DES"
            comp.note("generic 'DES' resolved to the paper's single DES family")


def load(rows, paper_key=""):
    """Rebuild a dictionary from its CSV rows (for a step run on its own)."""
    d = PaperDictionary(paper_key)
    for row in rows or []:
        if row.get("kind") == "alias":
            d.add_alias(row.get("code"), row.get("components"), row.get("source", ""),
                        row.get("evidence", ""))
            continue
        comp = Composition(
            written=row.get("code") or "",
            components=[c for c in str(row.get("components") or "").split(";") if c],
            components_written=[c for c in str(row.get("components_written") or "")
                                .split(";") if c],
            ratio_raw=str(row.get("ratio_raw") or ""),
            ratio_molar=str(row.get("ratio_molar") or ""))
        if comp.ratio_molar:
            try:
                comp.parts = [float(x) for x in comp.ratio_molar.split(":")]
            except ValueError:
                comp.parts = []
        wf = row.get("water_fraction")
        comp.water_fraction = float(wf) if wf not in (None, "") else None
        comp.water_basis = str(row.get("water_basis") or "")
        verified = str(row.get("verified")).lower() not in ("false", "0")
        d.add_mixture(row.get("code"), comp, row.get("source", ""), row.get("evidence", ""),
                      verified=verified)
    _set_family(d)
    return d
