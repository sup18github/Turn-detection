"""
Shared lexicon for Hinglish turn-detection heuristics.

These lists are intentionally romanized (Latin script) since most Hinglish ASR
transcripts (including AI4Bharat outputs) are produced or normalized in Roman
script for the English-mixed portions, and colloquial Hindi is very often
typed/transcribed in Roman script too. If you're working from Devanagari
transcripts, transliterate first or extend these lists with Devanagari forms.

Nothing here is exhaustive — treat as a starting point and expand from your
own data's error analysis (see README section 6, "Weak-label QA").
"""

# Words/sounds that typically mark hesitation, thinking, or a trailing filler.
# Presence at/near a pause boundary is a strong cue for INCOMPLETE_TURN.
FILLER_WORDS = {
    # pure hesitation sounds
    "umm", "uhmm", "uh", "um", "hmm", "hmmm", "ah", "aah", "err", "erm",
    # Hindi/Hinglish discourse fillers
    "arre", "are", "haan", "han", "achha", "acha", "matlab", "matlab_ki",
    "yaar", "toh", "to", "wo", "voh", "vo", "bas", "kya_kahen", "kya",
    "phir", "ek_min", "ek_second", "socho", "soch_raha_hoon", "soch_rahi_hoon",
    "kaise_kahoon", "kya_bataun", "iska_matlab", "ya_phir",
    # English fillers common in Hinglish speech
    "like", "actually", "basically", "you_know", "i_mean", "so_yeah",
    "well", "kind_of", "sort_of",
}

# Words that, when they appear right before or after a pause, strongly signal
# the clause is CONTINUING (i.e. the pause is not a turn boundary even if long).
CONTINUATION_WORDS = {
    "aur", "or", "and", "lekin", "but", "kyunki", "kyonki", "because",
    "ki", "that", "which", "jo", "jab", "when", "agar", "if", "toh", "then",
    "phir", "isliye", "so", "waise", "however", "although", "jabki",
    "while", "since", "taaki", "so_that", "warna", "otherwise",
}

# Words/punctuation-equivalents that mark genuine sentence/thought completion.
# In Roman-script ASR output there's usually no punctuation, so this is mostly
# used as a secondary cue alongside pause duration + F0 slope.
SENTENCE_FINAL_MARKERS = {
    "hai", "tha", "thi", "the", "hoon", "hain", "gaya", "gayi", "gaye",
    "diya", "diya_tha", "kar_diya", "ho_gaya", "ho_gayi", "theek_hai",
    "bas_itna_hi", "yehi_tha", "done", "okay", "ok", "thanks", "shukriya",
}


def normalize_token(token: str) -> str:
    """Lowercase + strip punctuation for lexicon lookups."""
    return token.strip().lower().strip(".,!?;:\u2013\u2014\"'")


def is_filler(token: str) -> bool:
    return normalize_token(token) in FILLER_WORDS


def is_continuation(token: str) -> bool:
    return normalize_token(token) in CONTINUATION_WORDS


def is_sentence_final(token: str) -> bool:
    return normalize_token(token) in SENTENCE_FINAL_MARKERS
