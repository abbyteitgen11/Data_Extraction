"""
Is a quote really in the paper? One answer, shared by every route that asks a model.

Two different tests were in use. Prose measurements required the exact quote, but
their status never looked at the result -- only at whether the NUMBER occurred
somewhere in the section -- so 10 of 14 accepted rows had a quote that was not in the
paper. Prose applications compared only the first 80 characters. Neither tolerated the
small rewrites a model makes when copying (a dropped space, a "–" for "-", a citation
it appended), so a faithful quote could fail while a fabricated one passed.

Here a quote is found when, after normalising typography, it occurs exactly or a window
of the text matches it at >= 90% similarity. Short quotes must match exactly: at five
words a 90% match is one changed word, which can be the number itself.
"""
import difflib
import re

_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-", "‐": "-", "‑": "-", "‒": "-",
                         "’": "'", "‘": "'", "“": '"', "”": '"', "·": ".", "⋅": ".",
                         "×": "x", " ": " ", " ": " ", " ": " "})
_CITATION = re.compile(r"\[\s*\d[\d,\s\-–]*\]")


def normalise(text):
    """Typography out, words in: lower case, one kind of dash, single spaces."""
    text = str(text or "").translate(_DASHES).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def _without_citations(text):
    return re.sub(r"\s+", " ", _CITATION.sub(" ", text)).strip()


def quote_found(quote, text, threshold=0.9):
    """Does `quote` occur in `text`, allowing for copying noise? -> bool."""
    q = normalise(quote).strip(" .…\"'")
    t = normalise(text)
    if not q or not t:
        return False
    if q in t:
        return True
    q2, t2 = _without_citations(q), _without_citations(t)
    if q2 and q2 in t2:
        return True
    if len(q2.split()) < 8:
        return False
    return best_similarity(q2, t2) >= threshold


def best_similarity(quote, text):
    """The best SequenceMatcher ratio of `quote` against any window of `text`."""
    n = len(quote)
    if n == 0 or not text:
        return 0.0
    stride = max(1, n // 6)
    best = 0.0
    for size in (n, int(n * 0.9), int(n * 1.1)):
        if size <= 0:
            continue
        for start in range(0, max(1, len(text) - size + 1), stride):
            window = text[start:start + size]
            matcher = difflib.SequenceMatcher(None, quote, window, autojunk=False)
            if matcher.quick_ratio() < best or matcher.quick_ratio() < 0.75:
                continue
            ratio = matcher.ratio()
            if ratio > best:
                best = ratio
                if best >= 0.98:
                    return best
    return best


def locate(quote, text):
    """The span of `text` the quote came from, for showing a human. -> str."""
    q, t = normalise(quote), normalise(text)
    i = t.find(q)
    if i >= 0:
        return t[i:i + len(q)]
    return ""


def _normalize_number(s):
    s = s.strip()
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def numbers_in(text):
    """Every number written in `text`, normalised so 1.20 == 1.2. -> set[str]."""
    text = str(text or "").translate(_DASHES)
    # Thousands separators: "1,548" is one number. Only a comma followed by exactly
    # three digits and then a non-digit is one.
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return {_normalize_number(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)}


def number_in(value, text):
    """Does this value occur as a number in `text`? -> bool."""
    if value is None:
        return False
    forms = {_normalize_number(f"{value}"), _normalize_number(f"{value:g}")}
    found = numbers_in(text)
    if forms & found:
        return True
    # "-79.2" written with a unicode minus that the model dropped, or the reverse.
    return any(f.lstrip("-") in {x.lstrip("-") for x in found} for f in forms)
