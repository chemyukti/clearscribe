"""Runtime configuration. Every value can be overridden with an environment variable."""

import os

# Speech model. Defaults to the public checkpoint used in the training notebook; point this at
# the noise-robust finetuned checkpoint (e.g. ".../wav2vec2_librispeech/final_model") to use it.
MODEL_NAME = os.environ.get("CLEARSCRIBE_MODEL", "facebook/wav2vec2-base-960h")

SAMPLE_RATE = 16000

# Long recordings are scored in overlapping windows so words at a cut keep their context.
CHUNK_SECONDS = float(os.environ.get("CLEARSCRIBE_CHUNK_SECONDS", 20))
CHUNK_OVERLAP_SECONDS = float(os.environ.get("CLEARSCRIBE_CHUNK_OVERLAP_SECONDS", 2))

# Beam decoder.
BEAM_WIDTH = int(os.environ.get("CLEARSCRIBE_BEAM_WIDTH", 16))
# Characters whose log-probability at a frame is below this are not expanded.
PRUNE_LOGP = float(os.environ.get("CLEARSCRIBE_PRUNE_LOGP", -12.0))

# Glossary boost, in log-probability units added per matched character of a glossary term.
# The bonus is withdrawn again if the partial match does not complete the term.
GLOSSARY_BOOST = float(os.environ.get("CLEARSCRIBE_GLOSSARY_BOOST", 1.5))
GLOSSARY_BOOST_MIN = 0.0
GLOSSARY_BOOST_MAX = 5.0

MAX_GLOSSARY_TERMS = 20
MAX_TERM_WORDS = 5
MAX_TERM_CHARS = 60

# Words whose confidence (min posterior over their emitting frames) is below this are flagged.
LOW_CONFIDENCE = float(os.environ.get("CLEARSCRIBE_LOW_CONFIDENCE", 0.5))

# Seconds of audio played before and after a suggestion when the reviewer presses replay (O4).
REPLAY_PADDING_SECONDS = 1.0

MAX_UPLOAD_MB = int(os.environ.get("CLEARSCRIBE_MAX_UPLOAD_MB", 50))
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
