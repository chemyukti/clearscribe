# ClearScribe

Glossary-assisted transcription review for noisy English interviews.

Before uploading a recording, the interviewer can list up to 20 names, places or short phrases
they expect to hear (for example the speaker's name or the street where the interview took place).
ClearScribe scores the audio once with the existing wav2vec 2.0 CTC model, then decodes it twice:

1. **Standard**: CTC prefix beam search with no glossary.
2. **Glossary-assisted**: the same beam search with a small boost for glossary terms.

Where the glossary-assisted transcript contains a glossary term and the standard transcript says
something else, the change goes to a **review queue** with timestamps. The reviewer replays that
part of the audio (±1 s around the words) and accepts or rejects it. Nothing in the transcript
changes until a suggestion is accepted. Glossary terms that never appear are reported as
*not heard* and stay out of the transcript.

Example: a noisy clip says "We met near Neeladri Road", and the glossary is `Neeladri Road`, `Aniruddha`.

| | |
|---|---|
| Standard | we met near **neela road** |
| Suggestion (pending review) | neela road → **Neeladri Road**, 0:03.2–0:04.1, ▶ replay |
| Glossary status | Neeladri Road: suggested · Aniruddha: not heard |

## Run the app

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The first transcription downloads and loads the model.

By default the app uses the public `facebook/wav2vec2-base-960h` checkpoint. To use the
noise-robust finetuned model, point `CLEARSCRIBE_MODEL` at its saved directory (the training
notebook writes it to `wav2vec2_librispeech/final_model`):

```bash
CLEARSCRIBE_MODEL=/path/to/wav2vec2_librispeech/final_model python app.py
```

Other settings (environment variables, see `clearscribe/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `CLEARSCRIBE_GLOSSARY_BOOST` | `1.5` | Default boost per matched character (the UI slider overrides it per upload, 0–5) |
| `CLEARSCRIBE_BEAM_WIDTH` | `16` | Beam width for both decodes |
| `CLEARSCRIBE_LOW_CONFIDENCE` | `0.5` | Words below this confidence get a dotted underline |
| `CLEARSCRIBE_CHUNK_SECONDS` | `20` | Long audio is scored in windows of this length, with 2 s of context on each side |
| `CLEARSCRIBE_UPLOAD_DIR` | system temp dir | Where uploaded audio is kept while the server runs |

## How the glossary boost works

`clearscribe/decoder.py` is a character-level CTC prefix beam search. Glossary terms are stored in a
character trie, and multi-word terms keep the word delimiter between their words. While a
hypothesis spells a term from a word boundary, each matching character adds `boost` to its
score. That bonus is **provisional**:

* If the hypothesis leaves the term mid-way ("NEELA" then a space), the bonus is withdrawn.
* The term only keeps its bonus when it is complete and followed by a word boundary (or the end of the audio).
* A term cannot start or end inside another word.

So the decoder is rewarded for spelling a whole glossary term, never for words that only start
like one. A term the speaker never said would need its characters to be emitted against the
acoustic evidence, which a small boost cannot outweigh. Both decodes use the same frame
log-probabilities, so every difference between them comes from the glossary.

Word timestamps come from the CTC frame where each character is emitted (20 ms per frame). Word
confidence is the minimum posterior of the word's characters at those frames, the same signal
used for word-level error flags. A word forced in by the glossary against weak evidence therefore
shows low confidence.

## Evaluate glossary off vs. on

Run the same held-out noisy clips with the glossary off and on:

```bash
python -m clearscribe.evaluate manifest.csv --out results/glossary_eval
```

`manifest.csv` needs `audio_path` and `text` columns. `audio_path` can point to wav/flac/ogg/mp3 or
to the `.pt` waveform tensors from the dataset pipeline. An optional `glossary` column holds terms
separated by `;`, and optional `noise_type`/`snr_db` columns break the results down by condition.
LibriSpeech has no glossaries, so `--auto-glossary` builds one per utterance: rare words from its
own reference (spoken terms) plus rare words from other utterances (unspoken terms). The choice
depends only on the reference text, so every noise condition of an utterance gets the same glossary.

Reported for glossary **off** and **on** (*on* means every suggestion accepted):

* **spoken_term_accuracy**: share of spoken glossary-term occurrences transcribed correctly
* **unspoken_term_rate**: share of (clip, unspoken term) pairs where the term appears in the output
* **wer**: word error rate

Results are written to `summary.json` (overall and per noise condition) and `per_clip.csv`.

## Tests

```bash
python -m pytest
```

The tests use synthetic CTC posteriors in the model's vocabulary layout and a stub model, so they
run without downloading the network.

## Layout

```
app.py                    Flask routes (upload, job, audio, review, export)
clearscribe/
  config.py               settings
  audio.py                load any audio file or .pt tensor as 16 kHz mono
  asr.py                  wav2vec 2.0 scoring, chunked for long recordings
  decoder.py              CTC prefix beam search + glossary boost, word timings and confidence
  glossary.py             glossary parsing, transcript comparison, review queue
  pipeline.py             score once, decode twice, build suggestions
  evaluate.py             glossary off/on evaluation
templates/index.html      page
static/app.js, style.css  frontend
tests/                    pytest suite
```

## Scope

No model retraining, speaker identification or multilingual support. The glossary applies to one
upload only and is not stored across recordings.
