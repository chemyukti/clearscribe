import pytest

from clearscribe.decoder import Word
from clearscribe.glossary import (GlossaryError, build_suggestions, final_segments, final_text,
                                  parse_glossary, term_status)


def words(text, conf=0.9):
    return [Word(w, i * 0.5, i * 0.5 + 0.4, conf) for i, w in enumerate(text.split())]


def test_parse_normalizes_and_deduplicates():
    terms = parse_glossary("Neeladri Road\n  aniruddha \nNEELADRI   road\n\n")
    assert [t.normalized for t in terms] == ["NEELADRI ROAD", "ANIRUDDHA"]
    assert terms[0].display == "Neeladri Road"


def test_parse_limits():
    assert len(parse_glossary([f"Term{chr(65 + i)}" for i in range(20)])) == 20
    with pytest.raises(GlossaryError, match="at most 20"):
        parse_glossary([f"Term{chr(65 + i)}" for i in range(21)])
    with pytest.raises(GlossaryError, match="no English letters"):
        parse_glossary(["12345"])
    with pytest.raises(GlossaryError, match="too long"):
        parse_glossary(["one two three four five six"])


def test_replacement_suggestion_uses_user_spelling_and_timestamps():
    terms = parse_glossary(["Neeladri Road", "Aniruddha"])
    std = words("WE MET NEAR NEELA ROAD")
    boosted = words("WE MET NEAR NEELADRI ROAD")
    (s,) = build_suggestions(std, boosted, terms, duration=10.0)
    assert (s["standard_text"], s["suggested_text"]) == ("neela road", "Neeladri Road")
    assert (s["a_start"], s["a_end"]) == (3, 5)
    assert s["start"] == 1.5 and s["end"] == 2.4
    assert s["replay_start"] == 0.5 and s["replay_end"] == 3.4
    assert s["status"] == "pending"

    status = {r["term"]: r["status"] for r in term_status(terms, std, boosted, [s])}
    assert status == {"Neeladri Road": "suggested", "Aniruddha": "not_heard"}


def test_insertion_and_unchanged_terms():
    terms = parse_glossary(["Aniruddha", "Road"])
    std = words("WE MET NEAR ROAD")
    boosted = words("WE MET ANIRUDDHA NEAR ROAD")
    (s,) = build_suggestions(std, boosted, terms, duration=3.0)
    assert (s["standard_text"], s["suggested_text"]) == ("", "Aniruddha")
    assert s["a_start"] == s["a_end"] == 2
    status = {r["term"]: r["status"] for r in term_status(terms, std, boosted, [s])}
    assert status["Road"] == "in_standard"


def test_suggestions_change_transcript_only_after_acceptance():
    terms = parse_glossary(["Neeladri Road"])
    std = words("WE MET NEAR NEELA ROAD")
    std[0].confidence = 0.2
    suggestions = build_suggestions(std, words("WE MET NEAR NEELADRI ROAD"), terms, 10.0)
    segments = final_segments(std, suggestions, low_confidence=0.5)
    assert segments[0]["low_confidence"] and not segments[1]["low_confidence"]

    assert final_text(segments, suggestions) == "we met near neela road"
    suggestions[0]["status"] = "rejected"
    assert final_text(segments, suggestions) == "we met near neela road"
    suggestions[0]["status"] = "accepted"
    assert final_text(segments, suggestions) == "we met near Neeladri Road"


def test_no_glossary_no_suggestions():
    std = words("HELLO THERE")
    assert build_suggestions(std, std, [], 2.0) == []
