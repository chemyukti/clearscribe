"""CTC prefix beam search with an optional glossary boost.

The decoder works on characters. Glossary terms are stored in a character trie (multi-word terms use
the word delimiter between words). While a hypothesis is spelling a glossary term from a word
boundary, each matching character earns ``boost`` (log-probability units). The bonus is provisional:
if the hypothesis leaves the term before finishing it, or finishes it without a word boundary
after it, the provisional bonus is withdrawn. So a hypothesis is only rewarded for a complete
glossary term, never for words that merely start like one ("NEELA" does not keep any bonus earned
on the way towards "NEELADRI").

With an empty glossary (or ``boost == 0``) this is a plain CTC prefix beam search, which gives the
"standard" transcript.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .config import BEAM_WIDTH, GLOSSARY_BOOST, PRUNE_LOGP

NEG_INF = -math.inf
MAX_CANDIDATES_PER_FRAME = 12


def _lae(a: float, b: float) -> float:
    """log(exp(a) + exp(b))"""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class _TrieNode:
    __slots__ = ("children", "is_term_end")

    def __init__(self):
        self.children = {}
        self.is_term_end = False


class GlossaryBias:
    """Scores a hypothesis prefix by how much of a glossary term it is spelling.

    State is an immutable tuple ``(node, provisional, committed)``:
      node        -- current trie node; ``None`` while inside a word that cannot start a term
      provisional -- bonus earned by the term currently being spelled (lost if it is not finished)
      committed   -- bonus from glossary terms that were completed and closed by a word boundary
    """

    def __init__(self, terms, delimiter: str, boost: float):
        self.delimiter = delimiter
        self.boost = boost
        self.root = _TrieNode()
        for term in terms:
            node = self.root
            for ch in term.replace(" ", delimiter):
                node = node.children.setdefault(ch, _TrieNode())
            node.is_term_end = True

    def initial(self):
        return (self.root, 0.0, 0.0)  # an utterance starts at a word boundary

    def advance(self, state, ch: str):
        node, provisional, committed = state
        if node is None:  # inside a non-glossary word: wait for the next word boundary
            return (self.root, 0.0, committed) if ch == self.delimiter else state

        if ch == self.delimiter:
            if node.is_term_end:  # a whole term followed by a word boundary: keep the bonus
                committed += provisional
                provisional = 0.0
            child = node.children.get(ch)
            if child is not None:  # the delimiter inside a multi-word term
                return (child, provisional + self.boost, committed)
            return (self.root, 0.0, committed)  # unfinished partial terms lose their bonus

        child = node.children.get(ch)
        if child is not None:
            return (child, provisional + self.boost, committed)
        return (None, 0.0, committed)  # left the term mid-word: withdraw the provisional bonus

    @staticmethod
    def running_score(state) -> float:
        return state[1] + state[2]

    @staticmethod
    def final_score(state) -> float:
        node, provisional, committed = state
        if node is not None and node.is_term_end:  # end of audio is a word boundary
            return committed + provisional
        return committed


class _NoBias:
    def initial(self):
        return None

    def advance(self, state, ch):
        return None

    @staticmethod
    def running_score(state):
        return 0.0

    final_score = running_score


@dataclass
class Word:
    text: str
    start: float          # seconds
    end: float            # seconds
    confidence: float     # min posterior of the word's characters at their emitting frames

    def to_dict(self):
        return {"text": self.text, "start": round(self.start, 3), "end": round(self.end, 3),
                "confidence": round(self.confidence, 4)}


@dataclass
class Decoded:
    text: str
    words: list = field(default_factory=list)


def beam_decode(log_probs: np.ndarray, vocab, frame_seconds: float, glossary_terms=(),
                boost: float = GLOSSARY_BOOST, beam_width: int = BEAM_WIDTH,
                prune_logp: float = PRUNE_LOGP) -> Decoded:
    """Decode (frames, vocab) CTC log-probabilities into words with timings and confidences.

    ``glossary_terms`` are normalized strings (upper case, words separated by spaces).
    """
    labels = vocab.labels
    blank = vocab.blank_id
    delim = labels[vocab.delimiter_id]
    usable = np.array([i for i, lab in enumerate(labels)
                       if i != blank and i not in vocab.ignored_ids and len(lab) == 1])
    char_id = {labels[i]: int(i) for i in usable}

    bias = GlossaryBias(glossary_terms, delim, boost) if glossary_terms and boost > 0 else _NoBias()

    # prefix -> [p_blank, p_non_blank]; meta: prefix -> (bias state, emitting frame of each char)
    beams = {"": [0.0, NEG_INF]}
    meta = {"": (bias.initial(), ())}

    for t in range(log_probs.shape[0]):
        lp = log_probs[t]
        lp_usable = lp[usable]
        keep = np.nonzero(lp_usable >= prune_logp)[0]
        if len(keep) > MAX_CANDIDATES_PER_FRAME:
            keep = keep[np.argsort(-lp_usable[keep])[:MAX_CANDIDATES_PER_FRAME]]
        candidates = [(labels[usable[k]], float(lp_usable[k])) for k in keep]
        lp_blank = float(lp[blank])

        nxt = {}
        nxt_meta = {}
        for prefix, (pb, pnb) in beams.items():
            total = _lae(pb, pnb)
            last = prefix[-1] if prefix else None

            entry = nxt.get(prefix)
            if entry is None:
                entry = nxt[prefix] = [NEG_INF, NEG_INF]
                nxt_meta[prefix] = meta[prefix]
            entry[0] = _lae(entry[0], total + lp_blank)
            if last is not None:  # a repeated character without a blank collapses
                entry[1] = _lae(entry[1], pnb + float(lp[char_id[last]]))

            for ch, lpc in candidates:
                if ch == delim and (last is None or last == delim):
                    # leading or doubled word delimiters add nothing to the text
                    if last is None:
                        entry[0] = _lae(entry[0], total + lpc)
                    else:
                        entry[1] = _lae(entry[1], pb + lpc)
                    continue
                new = prefix + ch
                score = (pb if ch == last else total) + lpc
                new_entry = nxt.get(new)
                if new_entry is None:
                    new_entry = nxt[new] = [NEG_INF, NEG_INF]
                    if new in meta:
                        nxt_meta[new] = meta[new]
                    else:
                        state, frames = meta[prefix]
                        nxt_meta[new] = (bias.advance(state, ch), frames + (t,))
                new_entry[1] = _lae(new_entry[1], score)

        ranked = sorted(nxt.items(),
                        key=lambda kv: _lae(*kv[1]) + bias.running_score(nxt_meta[kv[0]][0]),
                        reverse=True)[:beam_width]
        beams = dict(ranked)
        meta = {p: nxt_meta[p] for p in beams}

    best = max(beams, key=lambda p: _lae(*beams[p]) + bias.final_score(meta[p][0]))
    return _to_words(best, meta[best][1], log_probs, char_id, delim, frame_seconds)


def _to_words(prefix, frames, log_probs, char_id, delim, frame_seconds) -> Decoded:
    words, chars, char_frames = [], [], []

    def flush():
        if chars:
            conf = min(math.exp(float(log_probs[f, char_id[c]])) for c, f in zip(chars, char_frames))
            words.append(Word("".join(chars), char_frames[0] * frame_seconds,
                              (char_frames[-1] + 1) * frame_seconds, conf))
            chars.clear()
            char_frames.clear()

    for ch, f in zip(prefix, frames):
        if ch == delim:
            flush()
        else:
            chars.append(ch)
            char_frames.append(f)
    flush()
    return Decoded(" ".join(w.text for w in words), words)
