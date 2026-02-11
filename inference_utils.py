import numpy as np
import re

# Regex & word lists
slur_re = re.compile(r"\b(bitch|hoe|slut|whore|thot|milf|skank|cunt|roastie|feminazi)\b", re.I)
poss_re = re.compile(r"\b(my|our|his|her|their|your)\b", re.I)
gender_words = ["woman","women","girl","female","wife","lady","sister","daughter","girlfriend"]

# Helper functions
def _normalize_text(t):
    return str(t).lower().strip()

def has_gender_near_slur(text):
    tl = text.lower()
    g = "|".join(gender_words)
    s = "bitch|hoe|slut|whore|cunt|thot|roastie|feminazi|milf"
    pat1 = rf"\b({g})\b(?:\W+\w+){{0,6}}\W+({s})\b"
    pat2 = rf"\b({s})\b(?:\W+\w+){{0,6}}\W+({g})\b"
    return bool(re.search(pat1, tl) or re.search(pat2, tl))

def is_reclaimed_like(text):
    t = text.lower()
    strong = ["my ", "im a", "i'm a", "proud ", "queen", "badass", "boss",
              "cool ", "love being", "yes i am"]
    if any(w in t for w in strong) and any(sl in t for sl in ["bitch","hoe","slut","whore","cunt"]):
        return True
    if poss_re.search(t) and slur_re.search(t):
        return True
    if has_gender_near_slur(t):
        return True
    return False

def contains_obfuscated_slur(text: str) -> bool:
    """
    Checks if a text contains gender-specific slurs using highly flexible
    regex patterns to detect censorship, leetspeak, spacing, and truncation.
    """
    if not text or not isinstance(text, str):
        return False
    t = text.lower()

    # Core slurs used for pattern generation
    SLURS = ["bitch","whore","cunt","slut","thot","hoe","skank","roastie","feminazi","milf"]

    # Define a highly permissive filler for gaps and separators (0 or more non-word/non-space characters)
    FILLER = r"[\W_]*"

    # IMMEDIATE/SIMPLE OBFUSCATION CHECK
    simple_obfuscations = [
        "b!tch", "b1tch", "biitch", "b17ch", "b!7ch", "b*tch", "b***h", "b****",
        "h0e", "h03", "h**e", "h***",
        "slvt", "s!ut", "sl0t", "sl**t", "s***t",
        "wh0re", "wh0r3", "wh**e", "w***e", "whorrr",
        "c*nt", "c**nt", "c***t", "c****", "c.u.n.t", "c-u-n-t",
        "th0t", "th07", "th**t"
    ]
    if any(obf in t for obf in simple_obfuscations):
        return True

    # TRUNCATED CENSORSHIP
    # Checks for the first letter followed by 1 to 5 censorship characters, ensuring start-of-word match.
    truncated_censored_patterns = [
        r"\bc[\*\@\#\!\$\%\^\&\-\_\.]{1,5}",    # e.g., c*** (word boundary followed by c, then filler)
        r"\bs[\*\@\#\!\%\._\-]{1,5}",          # e.g., s***
        r"\bw[\*\#\@\!\%\._\-]{1,5}",          # e.g., w***
        r"\bb[\*\#\@\!\%\._\-]{1,5}",          # e.g., b****
    ]
    if any(re.search(pat, t) for pat in truncated_censored_patterns):
        return True

    for slur in SLURS:
        # GAP/MIXED SEPARATORS
        # Matches every letter separated by ANY filler characters.
        gap_pattern = FILLER.join(re.escape(c) for c in slur)
        if re.search(gap_pattern, t):
            return True

        # CENSORSHIP
        # Matches the first and last letter separated by 1 or more specific filler chars.
        censored_pattern = rf"{re.escape(slur[0])}[\*\@\#\!\$\%\^\&\-\_\.]{{1,}}{re.escape(slur[-1])}"
        if re.search(censored_pattern, t):
            return True

        # REPETITION
        # Matches if any single letter is repeated 3 or more times (e.g., b+i+t+c+h+).
        rep_pattern = "".join(rf"{c}+" for c in slur)
        if re.search(rep_pattern, t):
             return True

        # FLEXIBLE LEETSPEAK
        leet_substitutions = {
            'a': '[a@4]', 'e': '[e3]', 'i': '[i!1|]', 'o': '[o0]', 's': '[s5]', 't': '[t7+]', 'l': '[l1|]', 'h': '[h#]'
        }
        flexible_leet_slur = "".join(leet_substitutions.get(c, re.escape(c)) for c in slur)

        # Combine leetspeak with possible gaps/fillers
        leet_gap_pattern = FILLER.join(flexible_leet_slur)

        if re.search(leet_gap_pattern, t):
            return True

    return False


# Feature extraction
def extract_features_single(text):
    t = str(text)
    tl = t.lower()
    reclaimed_flag = int(is_reclaimed_like(tl))
    slur_flag = int(bool(slur_re.search(tl)))
    inter_flag = reclaimed_flag * slur_flag

    caps_ratio = sum(c.isupper() for c in t) / max(1, len(t))
    punctuation_density = len(re.findall(r"[.,;:\-_]", t)) / max(1, len(t))
    emoji_count = t.count("🔥") + t.count("💦") + t.count("🍑") + t.count("👅") + t.count("😈") + t.count("🤬")

    return np.array([
        reclaimed_flag,
        inter_flag,
        int("bitch" in tl), int("hoe" in tl), int("slut" in tl),
        int("whore" in tl), int("cunt" in tl),
        int("thot" in tl), int("roastie" in tl), int("feminazi" in tl),
        int(has_gender_near_slur(tl)),
        caps_ratio,
        len(re.findall(r"[!?.]{2,}", t)),
        emoji_count,
        punctuation_density
    ], dtype=float)

def extract_features_list(texts):
    return np.vstack([extract_features_single(t) for t in texts])

def extract_possessive_flag(texts):
    return np.array([1 if poss_re.search(str(t)) else 0 for t in texts]).reshape(-1,1)

def ensure_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1,1)
    return a


# Combined prediction logic
def final_prediction(text, sbert, model, global_thr, reclaimed_thr):
    tl = _normalize_text(text)

    # SBERT embedding
    emb = sbert.encode([tl], convert_to_numpy=True).astype("float32")
    hf  = extract_features_list([text]).astype("float32")
    pos = extract_possessive_flag([text]).astype("float32")
    X = np.hstack([ensure_2d(emb), ensure_2d(hf), ensure_2d(pos)])

    proba = float(model.predict_proba(X)[0][1])

    # Rule overrides
    if contains_obfuscated_slur(text):
        return proba, "SEXIST", "Obfuscated slur override"

    if is_reclaimed_like(tl) and slur_re.search(tl):
        return proba, "CLEAN", "Reclaimed positive usage override"

    if is_reclaimed_like(tl):
        if proba >= reclaimed_thr:
            return proba, "SEXIST", "Reclaimed-case threshold"
        else:
            return proba, "CLEAN", "Reclaimed-case threshold"

    # Normal threshold
    if proba >= global_thr:
        return proba, "SEXIST", "Global threshold"
    else:
        return proba, "CLEAN", "Global threshold"
