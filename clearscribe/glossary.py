"""Glossary parsing, transcript comparison, and the review queue.

The standard transcript is the one the reviewer starts from. A glossary-assisted change becomes a
*suggestion* only where the glossary-assisted transcript contains a glossary term and the standard
transcript says something different at that point. Suggestions change the final transcript only
after the reviewer accepts them.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .config import MAX_GLOSSARY_TERMS, MAX_TERM_CHARS, MAX_TERM_WORDS, REPLAY_PADDING_SECONDS


def normalize(text: str) -> str:
    """Same normalization as the training transcripts: A-Z and apostrophes, single spaces."""
    text = re.sub(r"[^A-Z' ]", " ", str(text).upper())
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class GlossaryTerm:
    display: str      # as the user typed it, e.g. "Neeladri Road"
    normalized: str   # e.g. "NEELADRI ROAD"

    @property
    def words(self):
        return tuple(self.normalized.split())

    def display_words(self):
        shown = re.sub(r"[^A-Za-z' ]", " ", self.display).split()
        if len(shown) == len(self.words):
            return shown
        return [w.capitalize() for w in self.words]


class GlossaryError(ValueError):
    pass


def parse_glossary(raw) -> list:
    """Parse user input (newline/comma/semicolon separated, or a list) into glossary terms."""
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else re.split(r"[\n,;]+", str(raw))
    terms, seen, problems = [], set(), []
    for item in items:
        display = re.sub(r"\s+", " ", str(item)).strip()
        if not display:
            continue
        norm = normalize(display)
        if not norm:
            problems.append(f'"{display}" has no English letters the speech model can spell.')
            continue
        if len(display) > MAX_TERM_CHARS or len(norm.split()) > MAX_TERM_WORDS:
            problems.append(f'"{display}" is too long; use names or phrases of up to '
                            f"{MAX_TERM_WORDS} words.")
            continue
        if norm in seen:
            continue
        seen.add(norm)
        terms.append(GlossaryTerm(display, norm))
    if len(terms) > MAX_GLOSSARY_TERMS:
        problems.append(f"Enter at most {MAX_GLOSSARY_TERMS} glossary terms "
                        f"(you entered {len(terms)}).")
    if problems:
        raise GlossaryError(" ".join(problems))
    return terms


def count_occurrences(words, term_words) -> int:
    return len(find_occurrences(words, term_words))


def find_occurrences(words, term_words) -> list:
    """Start indices of non-overlapping occurrences of ``term_words`` in ``words``."""
    n, found, i = len(term_words), [], 0
    while i + n <= len(words):
        if tuple(words[i:i + n]) == tuple(term_words):
            found.append(i)
            i += n
        else:
            i += 1
    return found


def build_suggestions(standard_words, boosted_words, terms, duration: float) -> list:
    """Review-queue entries where the glossary-assisted transcript adds a glossary term.

    Both word lists hold decoder ``Word`` objects. Each suggestion replaces the standard words
    ``[a_start, a_end)`` with the glossary-assisted wording.
    """
    a = [w.text for w in standard_words]
    b = [w.text for w in boosted_words]
    opcodes = SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()

    # Glossary-term occurrences in the assisted transcript, as spans of `b`.
    spans = []
    for term in terms:
        for s in find_occurrences(b, term.words):
            spans.append([s, s + len(term.words), [(s, term)]])
    spans.sort(key=lambda x: x[0])

    # Widen each span to whole differing blocks so the replacement is self-contained,
    # then merge spans that touch the same words.
    widened = []
    for b0, b1, occ in spans:
        a_idx, a_pos = [], None
        for tag, i1, i2, j1, j2 in opcodes:
            overlaps = j1 < b1 and j2 > b0 if j2 > j1 else b0 < j1 < b1
            if not overlaps:
                continue
            if a_pos is None:
                a_pos = i1 + max(0, b0 - j1) if tag == "equal" else i1
            if tag == "equal":
                lo, hi = max(j1, b0), min(j2, b1)
                a_idx.extend(range(i1 + lo - j1, i1 + hi - j1))
            else:
                a_idx.extend(range(i1, i2))
                b0, b1 = min(b0, j1), max(b1, j2)
        if a_pos is None:  # term at the very end, after every opcode
            a_pos = len(a)
        a0, a1 = (min(a_idx), max(a_idx) + 1) if a_idx else (a_pos, a_pos)
        if widened and (b0 < widened[-1]["b"][1] or a0 < widened[-1]["a"][1]
                        or a0 == widened[-1]["a"][0]):
            last = widened[-1]
            last["a"] = [min(last["a"][0], a0), max(last["a"][1], a1)]
            last["b"] = [min(last["b"][0], b0), max(last["b"][1], b1)]
            last["occ"].extend(occ)
        else:
            widened.append({"a": [a0, a1], "b": [b0, b1], "occ": list(occ)})

    suggestions = []
    for span in widened:
        (a0, a1), (b0, b1) = span["a"], span["b"]
        if a[a0:a1] == b[b0:b1]:
            continue  # the standard transcript already has the term here
        shown = [w.lower() for w in b[b0:b1]]
        for start, term in span["occ"]:
            for k, word in enumerate(term.display_words()):
                if b0 <= start + k < b1:
                    shown[start + k - b0] = word
        source = boosted_words[b0:b1] or standard_words[a0:a1]
        start = min(w.start for w in source)
        end = max(w.end for w in source)
        suggestions.append({
            "id": f"s{len(suggestions) + 1}",
            "terms": sorted({t.display for _, t in span["occ"]}),
            "a_start": a0,
            "a_end": a1,
            "standard_text": " ".join(w.lower() for w in a[a0:a1]),
            "suggested_text": " ".join(shown),
            "start": round(start, 3),
            "end": round(end, 3),
            "replay_start": round(max(0.0, start - REPLAY_PADDING_SECONDS), 3),
            "replay_end": round(min(duration, end + REPLAY_PADDING_SECONDS), 3),
            "confidence": round(min(w.confidence for w in boosted_words[b0:b1]), 4)
            if b1 > b0 else None,
            "status": "pending",
        })
    return suggestions


def term_status(terms, standard_words, boosted_words, suggestions) -> list:
    """Per-term outcome shown next to the transcript."""
    a = [w.text for w in standard_words]
    b = [w.text for w in boosted_words]
    suggested = {t for s in suggestions for t in s["terms"]}
    rows = []
    for term in terms:
        if term.display in suggested:
            status = "suggested"
        elif count_occurrences(a, term.words):
            status = "in_standard"
        elif count_occurrences(b, term.words):
            status = "suggested"
        else:
            status = "not_heard"
        rows.append({"term": term.display, "status": status})
    return rows


def final_segments(standard_words, suggestions, low_confidence: float) -> list:
    """The reviewed transcript as a sequence of words and suggestions (in standard-word order)."""
    by_start = {s["a_start"]: s for s in suggestions}
    segments, i = [], 0
    while i <= len(standard_words):
        s = by_start.get(i)
        if s is not None:
            segments.append({"kind": "suggestion", "id": s["id"]})
            if s["a_end"] > i:
                i = s["a_end"]
                continue
        if i == len(standard_words):
            break
        w = standard_words[i]
        segments.append({"kind": "word", **w.to_dict(), "text": w.text.lower(),
                         "low_confidence": w.confidence < low_confidence})
        i += 1
    return segments


def final_text(segments, suggestions) -> str:
    by_id = {s["id"]: s for s in suggestions}
    out = []
    for seg in segments:
        if seg["kind"] == "word":
            out.append(seg["text"])
        else:
            s = by_id[seg["id"]]
            text = s["suggested_text"] if s["status"] == "accepted" else s["standard_text"]
            if text:
                out.append(text)
    return " ".join(out)
