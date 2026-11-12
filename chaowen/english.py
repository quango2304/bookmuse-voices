"""English words for Chao Wen, who only knows Mandarin sounds: each word's espeak-ng phonemes (the list frontend.py builds)
become the nearest Mandarin syllables, as a Chinese reader would say the word (iPhone → ai4 feng1, Google → gu3 ge1).
The front end's English table then holds pinyin symbols ([initial or Ø, final, tone] per syllable) instead of phonemes, so the
apps' code is unchanged: a word on the list gives its syllables, any other word is spelled with Chinese letter names.

    python english.py say iPhone Google NASA       # prints the syllables (after frontend.py build)
"""
import sys
from pathlib import Path

from frontend import DATA, INITIALS

# Letter names as mainland readers say them.
LETTERS = {
    "A": "ei1", "B": "bi4", "C": "xi4", "D": "di4", "E": "yi4", "F": "ai2 fu5", "G": "ji4", "H": "ai1 qi1", "I": "ai4",
    "J": "zhei4", "K": "kei4", "L": "ai2 lou5", "M": "ai2 mu5", "N": "en1", "O": "ou1", "P": "pi4", "Q": "qiu1", "R": "a4 er5",
    "S": "ai2 si5", "T": "ti4", "U": "you1", "V": "wei1", "W": "da2 bu5 liu5", "X": "ai2 ke5 si5", "Y": "wai4", "Z": "zei4",
}
# Words whose usual Chinese reading the rules can't reach.
WORDS = {
    "iphone": "ai4 feng1", "ipad": "ai4 pai4", "ok": "ou1 kei4", "okay": "ou1 kei4", "google": "gu3 ge1", "wifi": "wai4 fei1",
    "email": "yi1 mei4 er5", "app": "ai4 pu5", "apple": "ai4 po5", "facebook": "fei1 si5 bu4 ke5", "youtube": "you2 tu1 bu5",
    "hello": "ha1 lou2", "hi": "hai1", "bye": "bai1", "yes": "ye1 si5", "no": "nou1", "sorry": "suo4 rui4", "baby": "bei4 bi5",
    "party": "pai4 dui4", "cool": "ku4", "mr": "mi4 si5 te5", "mrs": "mi4 si5 si5", "ms": "mi4 si5",
}

_PY = {}  # pinyin syllable → [initial, final]


# Syllables Chinese readers use for English that no character has (K is "kei"); the model knows both symbols.
LOAN = {("k", "ei"), ("Ø", "ei"), ("zh", "ei")}


def _valid():
    """Every (initial, final) a character or phrase in the front end's tables has, and LOAN."""
    if not _PY:
        _PY.update({k: True for k in LOAN})
        for name in ("chars.tsv", "phrases.tsv"):
            for line in (DATA / name).read_text().splitlines():
                for s in line.split("\t")[1].split(" "):
                    if s != "-":
                        ini, fin, _ = s.split(",")
                        _PY[(ini, fin)] = True
    return _PY


def ipa():
    """espeak's phonemes per word (and '#A' … '#Z'), as frontend.py build wrote them."""
    out = {}
    for line in (DATA / "english_ipa.tsv").read_text().splitlines():
        k, v = line.split("\t")
        out[k] = v.split(" ")
    return out


def _split(py: str):
    """'feng1' → ['f', 'eng', '1'] (Ø for no initial), checked against Mandarin's syllables."""
    tone, body = py[-1], py[:-1]
    for ini in sorted(INITIALS, key=len, reverse=True):
        if body.startswith(ini) and (ini, body[len(ini):]) in _valid():
            return [ini, body[len(ini):], tone]
    assert ("Ø", body) in _valid(), py
    return ["Ø", body, tone]


def _syl(spec: str):
    return [_split(p) for p in spec.split()]


# --- espeak phonemes → Mandarin syllables -------------------------------------------------------------------------------
TWO = {("e", "ɪ"): "eɪ", ("a", "ɪ"): "aɪ", ("ɔ", "ɪ"): "ɔɪ", ("a", "ʊ"): "aʊ", ("o", "ʊ"): "oʊ", ("ɪ", "ə"): "ɪə", ("e", "ə"): "eə",
       ("ʊ", "ə"): "ʊə", ("t", "ʃ"): "tʃ", ("d", "ʒ"): "dʒ", ("ɛ", "ɹ"): "eə", ("ɑ", "ɹ"): "ɑɹ", ("ɔ", "ɹ"): "ɔɹ", ("ɪ", "ɹ"): "ɪə"}
# vowel → finals to try, best first (the first that makes a real syllable with the onset wins)
VOWEL = {
    "i": ["i", "ei"], "iː": ["i", "ei"], "ɪ": ["i", "ei"], "ᵻ": ["i", "ei"], "e": ["ai", "ei"], "ɛ": ["ai", "ei"], "æ": ["ai", "a"],
    "eɪ": ["ei", "ai"], "aɪ": ["ai"], "ɔɪ": ["ao"], "aʊ": ["ao"], "oʊ": ["ou", "o", "uo"], "a": ["a"], "ɑ": ["a"], "ɑː": ["a"],
    "ɑɹ": ["a"], "ʌ": ["a", "e"], "ɐ": ["a"], "ɒ": ["o", "uo", "ao"], "ɔ": ["o", "uo", "ao"], "ɔː": ["o", "uo", "ao"],
    "ɔɹ": ["o", "uo", "ao"], "o": ["o", "uo", "ou"], "ʊ": ["u"], "u": ["u", "ou"], "uː": ["u", "ou"], "ə": ["e", "a"],
    "ɚ": ["e", "er"], "ɜ": ["e", "er"], "ɜː": ["e", "er"], "ɪə": ["i", "ie"], "eə": ["ai"], "ʊə": ["u"],
}
EXTRA = {"ɪə": "er5", "eə": "er5", "ʊə": "er5"}  # a second, weak syllable after these
# consonant → initials to try (before i / ü readers move s, k, g, h … to x, q, j)
CONS = {
    "p": ["p"], "b": ["b"], "t": ["t"], "d": ["d"], "ɾ": ["d", "t"], "k": ["k", "q"], "ɡ": ["g", "j"], "g": ["g", "j"],
    "f": ["f"], "v": ["w", "f"], "θ": ["s", "x"], "ð": ["z", "d"], "s": ["s", "x"], "z": ["z", "j"], "ʃ": ["sh", "x"],
    "ʒ": ["r", "zh"], "tʃ": ["ch", "q"], "dʒ": ["zh", "j"], "h": ["h", "x"], "m": ["m"], "n": ["n"], "l": ["l"],
    "ɹ": ["r", "l"], "r": ["r", "l"], "j": ["y"], "w": ["w"], "ʔ": [],
}
# a consonant with no vowel of its own (clusters, word ends) gets a weak vowel
ALONE = {"p": "pu5", "b": "bu5", "t": "te5", "d": "de5", "ɾ": "de5", "k": "ke5", "ɡ": "ge5", "g": "ge5", "f": "fu5", "v": "fu5",
         "θ": "si5", "ð": "de5", "s": "si5", "z": "zi5", "ʃ": "shi5", "ʒ": "ri5", "tʃ": "qi5", "dʒ": "ji5", "h": "he5", "m": "mu5",
         "n": "en5", "l": "er5", "ɹ": "", "r": "", "ŋ": "", "j": "", "w": "", "ʔ": ""}



def phones(tokens):
    out, stress, i = [], None, 0
    toks = [t for t in tokens if t not in ("ː",) or True]
    while i < len(toks):
        t = toks[i]
        if t in ("ˈ", "ˌ"):
            stress = t; i += 1; continue
        if t == "ː":
            if out:
                out[-1] = (out[-1][0] + "ː" if out[-1][0] + "ː" in VOWEL else out[-1][0], out[-1][1])
            i += 1; continue
        if i + 1 < len(toks) and (t, toks[i + 1]) in TWO and (t, toks[i + 1]) != ("ɛ", "ɹ") or (i + 1 < len(toks) and (t, toks[i + 1]) == ("ɛ", "ɹ")):
            t = TWO[(t, toks[i + 1])]; i += 1
        if t in VOWEL:
            out.append((t, stress)); stress = None
        else:
            out.append((t, None))
        i += 1
    return out


def to_pinyin(tokens, tones=("1", "1")):
    """espeak tokens → list of [initial, final, tone]. tones = (stressed, unstressed)."""
    ph = phones(tokens)
    syl, i = [], 0
    valid = _valid()
    while i < len(ph):
        p, st = ph[i]
        if p in VOWEL:  # a vowel with no onset
            onset = None
        elif i + 1 < len(ph) and ph[i + 1][0] in VOWEL and p in CONS and CONS[p]:
            onset = p; i += 1; p, st = ph[i]
        else:
            if p in ("n", "ŋ", "m") and syl and syl[-1][1] in ("a", "e", "i", "u", "ia", "ua", "ie", "o", "uo"):
                pass  # handled as a coda below (never reached: codas are taken right after the vowel)
            a = ALONE.get(p, "")
            if a:
                syl.append(_split(a))
            i += 1
            continue
        tone = tones[0] if st else tones[1]
        # coda n / ŋ joins the final when the result is a real syllable
        coda = ph[i + 1][0] if i + 1 < len(ph) and ph[i + 1][0] in ("n", "ŋ") and not (i + 2 < len(ph) and ph[i + 2][0] in VOWEL) else ""
        best = None
        fins = VOWEL[p]
        if onset == "j" and p in ("u", "uː", "ʊ"):
            fins = ["ou"] + fins  # you, not yu (ü)
        inis = CONS[onset] if onset else ["Ø"]
        if p in ("i", "iː", "ɪ", "ᵻ", "ɪə"):
            # before i readers say x / j / q (si, zi, shi … are a different vowel in Mandarin), and r is l: Harry → ha li
            inis = {"s": ["x"], "θ": ["x"], "ʃ": ["x"], "z": ["j"], "dʒ": ["j"], "ʒ": ["j"], "tʃ": ["q"], "ɹ": ["l"], "r": ["l"]}.get(onset, inis)  # Harry → ha li, Paris → pa li si: r before i is l, as Chinese readers say it
        for fin in fins:
            for ini in inis:
                for f2 in ([fin + ("ng" if coda == "ŋ" else "n")] if coda else []) + [fin]:
                    if f2 == "ein" or f2 == "aing":
                        continue
                    for i2, fx in ((ini, f2),) if ini != "Ø" else (("Ø", f2), ("y", "i" + f2[1:] if f2.startswith("i") else f2), ("w", f2[1:] if f2.startswith("u") else f2)):
                        if (i2, fx) in valid:
                            best = [i2, fx, tone, f2 != fin]
                            break
                    if best:
                        break
                if best:
                    break
            if best:
                break
        if best is None:
            best = ["Ø", "a", tone, False]
        used_coda = best.pop()
        syl.append(best)
        if p in EXTRA:
            syl.append(_split(EXTRA[p]))
        i += 2 if used_coda else 1
    return syl


def word(w: str, english: dict, tones=("1", "1")):
    k = w.lower()
    if k in WORDS:
        return _syl(WORDS[k])
    if k in english:
        return to_pinyin(english[k], tones)
    return [s for ch in w if ch.isalpha() for s in _syl(LETTERS[ch.upper()])]


def table(tones=("1", "1")):
    """The English table for the pack: word → pinyin symbols ('Ø ai 4 f eng 1'), and '#A' … '#Z' for spelling."""
    eng = ipa(); out = {}
    bare = lambda ph: [p for p in ph if p not in ("ˈ", "ˌ", "ʲ")]
    letters = {ch: bare(eng["#" + ch]) for ch in LETTERS}
    for k, ph in eng.items():
        if k.startswith("#"):
            continue
        # a word espeak spells (USB, DNA) is spelled with Chinese letter names
        spelled = all(c.upper() in letters for c in k) and bare(ph) == [p for c in k for p in letters[c.upper()]]
        out[k] = [s for c in k for s in _syl(LETTERS[c.upper()])] if spelled else word(k, eng, tones)
    for k, v in WORDS.items():
        out[k] = _syl(v)
    for ch, v in LETTERS.items():
        out["#" + ch] = _syl(v)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "say":
        eng = ipa()
        for w in sys.argv[2:]:
            print(w, " ".join(f"{a if a != 'Ø' else ''}{b}{c}" for a, b, c in word(w, eng)), "  ", " ".join(eng.get(w.lower(), [])))
