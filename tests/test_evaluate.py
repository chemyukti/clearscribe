from clearscribe.evaluate import auto_glossaries, score_clip, summarize
from clearscribe.glossary import parse_glossary


def test_metrics_count_spoken_and_unspoken_terms():
    terms = parse_glossary(["Neeladri Road", "Aniruddha"])
    ref = "WE MET NEAR NEELADRI ROAD"
    off = score_clip(ref, "WE MET NEAR NEELA ROAD", terms)
    on = score_clip(ref, "WE MET NEAR NEELADRI ROAD", terms)
    bad = score_clip(ref, "WE MET ANIRUDDHA NEELADRI ROAD", terms)
    assert summarize([off])["spoken_term_accuracy"] == 0.0
    assert summarize([on]) == {"clips": 1, "spoken_term_occurrences": 1, "unspoken_term_pairs": 1,
                                    "spoken_term_accuracy": 1.0, "unspoken_term_rate": 0.0, "wer": 0.0}
    assert summarize([bad])["unspoken_term_rate"] == 1.0
    assert summarize([off])["wer"] == 0.2


def test_auto_glossary_is_stable_per_utterance():
    refs = ["THE MARCHIONESS ARRIVED AT DAWN", "BARTHOLOMEW SAILED NORTHWARD", "THE MARCHIONESS ARRIVED AT DAWN"]
    g = auto_glossaries(refs, spoken_per_clip=1, unspoken_per_clip=1)
    spoken, unspoken = g[refs[0]]
    assert spoken.upper() in refs[0].split()
    assert unspoken.upper() not in refs[0].split()
    assert g == auto_glossaries(refs, spoken_per_clip=1, unspoken_per_clip=1)
