"""
Post-processing for raw recognizer output.
Paper 1: WordNinja word segmentation, confidence bleed-through filtering.
"""
from typing import List, Tuple, Optional
import re


try:
    import wordninja
    _HAS_WORDNINJA = True
except ImportError:
    _HAS_WORDNINJA = False

try:
    from rapidfuzz.distance import Levenshtein as _Lev
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


def segment_words(text: str) -> List[str]:
    """
    Use WordNinja to split merged/concatenated text into dictionary words.
    Falls back to whitespace splitting when WordNinja is unavailable.
    """
    if not text:
        return []
    if _HAS_WORDNINJA:
        return wordninja.split(text)
    return text.split()


def filter_low_confidence(
    texts: List[str],
    confidences: List[float],
    threshold: float = 0.3,
    bleed_sigma: float = 0.15,
) -> List[Tuple[str, float]]:
    """
    Confidence bleed-through filtering.
    Paper 1: words with confidence below (block_avg - bleed_sigma * block_std)
    are discarded as bleed-through artefacts.

    Args:
        texts: list of recognised strings per word.
        confidences: list of CTC confidence scores in [0, 1].
        threshold: global hard-minimum confidence.
        bleed_sigma: number of std-devs below mean to gate bleed-through.

    Returns:
        Filtered list of (text, confidence) tuples.
    """
    if not texts:
        return []

    import numpy as np  # local import to keep module lightweight
    conf_arr = np.array(confidences)

    mu = conf_arr.mean()
    sigma = conf_arr.std()
    dynamic_thresh = max(threshold, mu - bleed_sigma * sigma)

    results = []
    for t, c in zip(texts, confidences):
        if c >= dynamic_thresh and t.strip():
            results.append((t, c))
    return results


def clean_ocr_text(text: str) -> str:
    """
    Basic text cleaning: strip trailing whitespace, normalise line-breaks,
    remove isolated punctuation runs.
    """
    # Normalise whitespace within lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def merge_hyphenated_words(text: str) -> str:
    """Merge hyphen-broken line-end words across line boundaries."""
    return re.sub(r"-\n(\S)", r"\1", text)


def correct_word_lexicon(word: str, lexicon: Optional[List[str]] = None) -> str:
    """Paper 1 Section III-F: vocabulary-guided lexicon prior correction."""
    if not word or not lexicon:
        return word
    w_clean = word.strip(".,!?\"'")
    if not w_clean:
        return word
    # Exact case-insensitive match
    for cand in lexicon:
        if w_clean.lower() == cand.lower():
            return cand

    # Do not mutate very short words (length < 3) through fuzzy edit distance
    if len(w_clean) < 3:
        return word

    best_cand = word
    best_dist = 999
    # Strict edit distance: max 1 edit for normal words, max 2 for words >= 6 chars
    max_d = 1 if len(w_clean) < 6 else 2

    for cand in lexicon:
        m, n = len(w_clean), len(cand)
        if abs(m - n) > max_d:
            continue
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                if w_clean[i - 1].lower() == cand[j - 1].lower():
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
            prev = curr
        d = prev[n]
        if d < best_dist and d <= max_d:
            best_dist = d
            best_cand = cand
    return best_cand


def postprocess_page_text(
    line_texts: List[str],
    line_confidences: List[float],
    do_wordninja: bool = True,
    lexicon: Optional[List[str]] = None,
    confidence_threshold: float = 0.2,
) -> Tuple[str, List[str]]:
    """
    Full postprocessing pipeline for a recognised page.

    Args:
        line_texts: list of raw OCR text per line.
        line_confidences: per-line average confidence.
        do_wordninja: whether to apply WordNinja word segmentation.
        lexicon: optional vocabulary list for lexicon correction.
        confidence_threshold: minimum confidence to include a line.

    Returns:
        (full_page_text, filtered_lines)
    """
    filtered = []
    lex_lower = {c.lower() for c in lexicon} if lexicon else set()

    for text, conf in zip(line_texts, line_confidences):
        if conf < confidence_threshold:
            continue
        text = clean_ocr_text(text)
        if not text:
            continue
        words = text.split()
        if lexicon:
            words = [correct_word_lexicon(w, lexicon) for w in words]
        if do_wordninja and len(words) == 1 and not any(c.isspace() for c in text):
            # Only segment if the concatenated token is long and not in lexicon
            if len(words[0]) >= 10 and words[0].lower() not in lex_lower:
                words = segment_words(words[0])
        # Filter isolated noise artifacts (punctuation and non-word single chars)
        valid_single = {'a', 'i', 'A', 'I'}
        good_words = []
        for w in words:
            w_strip = w.strip(".,!?\"':;~`_-()[]{}|")
            if not w_strip:
                continue
            if len(w_strip) == 1 and w_strip not in valid_single and not w_strip.isdigit():
                continue
            good_words.append(w)
        words = good_words
        if words:
            text = " ".join(words)
            filtered.append(text)

    full_text = "\n".join(filtered)
    full_text = merge_hyphenated_words(full_text)
    return full_text, filtered
