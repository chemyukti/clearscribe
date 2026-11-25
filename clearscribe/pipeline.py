"""Score once, decode twice (glossary off / on), and build the review queue."""

import time

from .config import GLOSSARY_BOOST, LOW_CONFIDENCE, SAMPLE_RATE
from .decoder import beam_decode
from .glossary import build_suggestions, final_segments, final_text, term_status


def transcribe(wave, terms, model, boost: float = GLOSSARY_BOOST) -> dict:
    duration = len(wave) / SAMPLE_RATE

    t0 = time.perf_counter()
    scores = model.score(wave)
    t1 = time.perf_counter()
    standard = beam_decode(scores.log_probs, scores.vocab, scores.frame_seconds)
    if terms and boost > 0:
        assisted = beam_decode(scores.log_probs, scores.vocab, scores.frame_seconds,
                               glossary_terms=[t.normalized for t in terms], boost=boost)
    else:
        assisted = standard
    t2 = time.perf_counter()

    suggestions = build_suggestions(standard.words, assisted.words, terms, duration)
    segments = final_segments(standard.words, suggestions, LOW_CONFIDENCE)
    return {
        "duration": round(duration, 3),
        "boost": boost,
        "glossary": [t.display for t in terms],
        "standard": {"text": standard.text.lower(), "words": [w.to_dict() for w in standard.words]},
        "assisted": {"text": assisted.text.lower(), "words": [w.to_dict() for w in assisted.words]},
        "suggestions": suggestions,
        "terms": term_status(terms, standard.words, assisted.words, suggestions),
        "segments": segments,
        "final_text": final_text(segments, suggestions),
        "timing": {
            "scoring_seconds": round(t1 - t0, 3),
            "decoding_seconds": round(t2 - t1, 3),
            "real_time_factor": round((t2 - t0) / duration, 3) if duration else None,
        },
    }
