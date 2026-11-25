"""Glossary off vs. on, on the same held-out noisy clips.

    python -m clearscribe.evaluate manifest.csv --out results/glossary_eval

The manifest is a CSV with columns
    audio_path   wav/flac/ogg/mp3, or a .pt waveform tensor from the dataset pipeline
    text         reference transcript
    glossary     (optional) terms separated by ";"
    noise_type, snr_db   (optional) used to break results down by condition

Without a ``glossary`` column, use ``--auto-glossary``: each clip gets rare words from its own
reference (spoken terms) plus rare words taken from other clips (unspoken terms). The selection
depends only on the reference text, so every noise condition of an utterance gets the same glossary.

Reported, with the glossary off and on (on = every glossary suggestion accepted):
    spoken_term_accuracy  share of spoken glossary-term occurrences transcribed correctly
    unspoken_term_rate    share of (clip, unspoken term) pairs where the term appears in the output
    wer                   word error rate
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .audio import load_audio
from .config import GLOSSARY_BOOST
from .decoder import beam_decode
from .glossary import count_occurrences, normalize, parse_glossary


def edit_distance(ref, hyp) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def auto_glossaries(references, spoken_per_clip=2, unspoken_per_clip=2, min_len=6, seed=42):
    """Pick glossary terms from the references: rare spoken words plus unspoken distractors."""
    unique = sorted(set(references))
    doc_freq = Counter(w for ref in unique for w in set(ref.split()))
    rare = sorted(w for w, n in doc_freq.items() if n == 1 and len(w) >= min_len)
    out = {}
    for ref in unique:
        rng = random.Random(f"{seed}:{ref}")
        words = set(ref.split())
        spoken = sorted(w for w in words if w in doc_freq and doc_freq[w] == 1 and len(w) >= min_len)
        spoken = rng.sample(spoken, min(spoken_per_clip, len(spoken)))
        pool = [w for w in rare if w not in words]
        unspoken = rng.sample(pool, min(unspoken_per_clip, len(pool)))
        out[ref] = [w.capitalize() for w in spoken + unspoken]
    return out


def score_clip(reference, hypothesis, terms):
    ref, hyp = reference.split(), hypothesis.split()
    row = {"spoken": 0, "spoken_correct": 0, "unspoken": 0, "unspoken_present": 0,
           "errors": edit_distance(ref, hyp), "ref_words": len(ref)}
    for term in terms:
        in_ref = count_occurrences(ref, term.words)
        in_hyp = count_occurrences(hyp, term.words)
        if in_ref:
            row["spoken"] += in_ref
            row["spoken_correct"] += min(in_ref, in_hyp)
        else:
            row["unspoken"] += 1
            row["unspoken_present"] += int(in_hyp > 0)
    return row


def summarize(rows):
    total = defaultdict(int)
    for r in rows:
        for k, v in r.items():
            if isinstance(v, int):
                total[k] += v
    ratio = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {
        "clips": len(rows),
        "spoken_term_occurrences": total["spoken"],
        "unspoken_term_pairs": total["unspoken"],
        "spoken_term_accuracy": ratio(total["spoken_correct"], total["spoken"]),
        "unspoken_term_rate": ratio(total["unspoken_present"], total["unspoken"]),
        "wer": ratio(total["errors"], total["ref_words"]),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--out", default="results/glossary_eval")
    p.add_argument("--boost", type=float, default=GLOSSARY_BOOST)
    p.add_argument("--auto-glossary", action="store_true")
    p.add_argument("--spoken-per-clip", type=int, default=2)
    p.add_argument("--unspoken-per-clip", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="only evaluate the first N clips")
    args = p.parse_args(argv)

    with open(args.manifest, newline="") as f:
        clips = list(csv.DictReader(f))[: args.limit]
    if not clips:
        sys.exit("The manifest has no rows.")
    for c in clips:
        c["reference"] = normalize(c["text"])

    if args.auto_glossary:
        chosen = auto_glossaries([c["reference"] for c in clips], args.spoken_per_clip,
                                 args.unspoken_per_clip)
        for c in clips:
            c["glossary"] = ";".join(chosen[c["reference"]])
    elif "glossary" not in clips[0]:
        sys.exit("The manifest has no 'glossary' column; add one or pass --auto-glossary.")

    from .asr import get_model
    model = get_model()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_clip = []
    for n, c in enumerate(clips, 1):
        terms = parse_glossary(c.get("glossary", "").split(";"))
        scores = model.score(load_audio(c["audio_path"]))
        off = beam_decode(scores.log_probs, scores.vocab, scores.frame_seconds).text
        on = beam_decode(scores.log_probs, scores.vocab, scores.frame_seconds,
                         glossary_terms=[t.normalized for t in terms], boost=args.boost).text
        for mode, hyp in (("off", off), ("on", on)):
            per_clip.append({"audio_path": c["audio_path"], "noise_type": c.get("noise_type", ""),
                             "snr_db": c.get("snr_db", ""), "glossary": c.get("glossary", ""),
                             "mode": mode, "reference": c["reference"], "hypothesis": hyp,
                             **score_clip(c["reference"], hyp, terms)})
        print(f"[{n}/{len(clips)}] {c['audio_path']}", file=sys.stderr)

    summary = {"boost": args.boost, "model": model.model_name, "overall": {}, "by_condition": {}}
    for mode in ("off", "on"):
        rows = [r for r in per_clip if r["mode"] == mode]
        summary["overall"][mode] = summarize(rows)
        groups = defaultdict(list)
        for r in rows:
            groups[f"{r['noise_type'] or 'all'} @ {r['snr_db'] or '-'}"].append(r)
        if len(groups) > 1:
            for key, g in sorted(groups.items()):
                summary["by_condition"].setdefault(key, {})[mode] = summarize(g)

    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_clip[0]))
        writer.writeheader()
        writer.writerows(per_clip)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    print(f"\n{'':24}{'glossary off':>14}{'glossary on':>14}")
    for key, label in (("spoken_term_accuracy", "spoken terms correct"),
                       ("unspoken_term_rate", "unspoken terms output"), ("wer", "WER")):
        print(f"{label:24}{pct(summary['overall']['off'][key]):>14}"
              f"{pct(summary['overall']['on'][key]):>14}")
    print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'per_clip.csv'}")


if __name__ == "__main__":
    main()
