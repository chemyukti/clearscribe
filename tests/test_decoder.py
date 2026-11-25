import pytest

from clearscribe.decoder import GlossaryBias, beam_decode
from conftest import FRAME_SECONDS, VOCAB, frames_to_log_probs, spell


def decode(log_probs, terms=(), boost=1.5):
    return beam_decode(log_probs, VOCAB, FRAME_SECONDS, glossary_terms=list(terms), boost=boost)


def test_plain_decoding_collapses_repeats_and_blanks():
    frames = ["W", "W", "-", "E", "|", "|", "-", "M", "E", "-", "E", "T", "T", "-"]
    assert decode(frames_to_log_probs(frames)).text == "WE MEET"


def test_standard_decode_misses_the_unfamiliar_name(neeladri_scores):
    assert decode(neeladri_scores.log_probs).text == "WE MET NEAR NEELA ROAD"


def test_glossary_recovers_spoken_name_and_leaves_unspoken_name_out(neeladri_scores):
    out = decode(neeladri_scores.log_probs, ["NEELADRI ROAD", "ANIRUDDHA"])
    assert out.text == "WE MET NEAR NEELADRI ROAD"
    assert "ANIRUDDHA" not in out.text


def test_no_boost_means_no_change(neeladri_scores):
    out = decode(neeladri_scores.log_probs, ["NEELADRI ROAD"], boost=0)
    assert out.text == "WE MET NEAR NEELA ROAD"


def test_unspoken_term_is_not_forced_into_clear_speech():
    lp = frames_to_log_probs(spell("WE MET HERE"))
    assert decode(lp, ["ANIRUDDHA", "NEELADRI ROAD"], boost=3.0).text == "WE MET HERE"


def test_word_timings_and_confidence(neeladri_scores):
    out = decode(neeladri_scores.log_probs, ["NEELADRI ROAD"])
    name = out.words[3]
    assert name.text == "NEELADRI"
    assert name.start == pytest.approx(len(spell("WE MET NEAR ")) * FRAME_SECONDS)
    assert name.end > name.start
    # D, R and I were only 30% likely; the word's confidence reflects that.
    assert name.confidence == pytest.approx(0.3, abs=0.02)
    assert out.words[0].confidence > 0.9


def _bias_after(text, terms, boost=1.0):
    bias = GlossaryBias(terms, "|", boost)
    state = bias.initial()
    for ch in text.replace(" ", "|"):
        state = bias.advance(state, ch)
    return bias, state


def test_partial_match_keeps_no_bonus():
    bias, state = _bias_after("NEELA ROAD", ["NEELADRI ROAD"])
    assert bias.final_score(state) == 0


def test_complete_term_keeps_bonus_including_word_delimiter():
    bias, state = _bias_after("NEAR NEELADRI ROAD", ["NEELADRI ROAD"])
    assert bias.final_score(state) == len("NEELADRI|ROAD")


def test_term_must_start_and_end_at_word_boundaries():
    for text in ("XNEELADRI", "NEELADRIS"):
        bias, state = _bias_after(text, ["NEELADRI"])
        assert bias.final_score(state) == 0
