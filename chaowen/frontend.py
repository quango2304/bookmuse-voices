#!/usr/bin/env python3
"""Chao Wen's text front end as the apps run it: the reference the Swift and Kotlin `ChineseText` ports are checked against.

Text → phoneme ids, and for every id the index of the character (in the text given) it came from:
1. NFKC; Traditional → Simplified (OpenCC t2s, longest match, same length only); numbers in words (`normalize`).
2. Chinese runs → pypinyin's readings: its forward longest match (`pypinyin.seg.mmseg`, no_non_phrases) over its phrases, each
   phrase's and character's (initial, final, tone) precomputed here (`build`) as Suxin's front end makes them from pypinyin
   (strict=False), with our fixes; then 一 / 不 tone changes.
3. Latin words → Mandarin syllables from a table made here (english.py) of the 100,000 most common English words; unknown
   words spelled with Chinese letter names.
4. Symbols → ids (the model's table): a pad after every tone, punctuation mark and space; BOS first, EOS last.

    uv run python frontend.py build            # writes work/chaowen/frontend/{phrases,chars,t2s,english}.tsv, phonemes.json
    uv run python frontend.py say "文本"        # prints the symbols and ids
    uv run python frontend.py golden FILE      # writes FILE.golden.json: ids and character indices per line, for the ports
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent / "work/chaowen"
DATA = WORK / "frontend"
CONFIG = WORK / "zh_CN-chaowen-medium.onnx.json"

# --- Suxin-Light's front end (suxin_tts/mixed_frontend.py), the parts we keep; Chao Wen's pinyin table is the same ---
HAN = re.compile("[\u3007\u4e00-\u9fff\u3400-\u4dbf]")  # 〇 too (二〇二六)
PUNCT = {"。": "。", ".": "。", "？": "？", "?": "？", "！": "！", "!": "！", "、": "，", "，": "，", ",": "，", "：": "，",
         ":": "，", "；": "，", ";": "，", "—": "，", "…": "，", "–": "，", "～": "，", "~": "，"}
GROUP_END = {"1", "2", "3", "4", "5", "。", "？", "！", "，", " ", ",", ".", "?", "!", ";", ":"}
FINAL_ALIAS = {"ü": "v", "üe": "ve", "üan": "van", "ün": "vn", "ve": "ve", "van": "van", "vn": "vn", "ue": "ue", "v": "v"}
INITIALS = set("b p m f d t n l g k h j q x zh ch sh r z c s y w".split())
FINALS = set("a o e ai ei ao ou an en ang eng ong i ia ie iao iu ian in iang ing iong u ua uo uai ui uan un uang ueng v ve van vn "
             "er ue".split())


def syllable(ini: str, fin_tone: str):
    """pypinyin's initial and final+tone → [initial or Ø, final, tone 1–5], as Suxin's `_zh_syllable`; None when unusable."""
    tone, fin = "", fin_tone
    if fin and fin[-1].isdigit():
        tone, fin = fin[-1], fin[:-1]
    if not fin:
        return None
    fin = FINAL_ALIAS.get(fin, fin)
    if fin not in FINALS and fin not in ("n", "ng"):
        return None
    return [ini if ini in INITIALS else "Ø", fin, tone or "5"]


# --- Our fixes on top of pypinyin ----------------------------------------------------------------------------------------
# A character standing alone (not inside one of pypinyin's phrases): its usual reading in running text.
CHAR_FIX = {"长": ["ch", "ang", "2"], "得": ["d", "e", "5"], "似": ["s", "i", "4"]}
# Suxin's own (it applies inside phrases too).
ALWAYS = {"血": ["x", "ie", "3"]}
PHRASE_ADD = {"长得": [["zh", "ang", "3"], ["d", "e", "5"]]}
# 地 after an adverb is the particle de: after a doubled character (慢慢地) or one of these endings (突然地, 猛地, 认真地 …).
DE_AFTER = set("然猛声真断意力般样默悄渐轻慢静紧深狠远微缓细快心劲情")
NUMERAL = set("零一二三四五六七八九十百千万亿两点第")
# 两 rather than 二 for a quantity of 2 before these.
MEASURE = set("个只本位次天年件种条张把台辆句点分秒小周岁层名家双对份步层页章块棵头口斤")

DIGITS = "零一二三四五六七八九"
# A digit no rule read (rare after NFKC): its reading by value, as (initial, final+tone) for `syllable`.
LEFTOVER_DIGIT = [("l", "ing2"), ("y", "i1"), ("", "er4"), ("s", "an1"), ("s", "i4"), ("", "u3"), ("l", "iu4"), ("q", "i1"),
                  ("b", "a1"), ("j", "iu3")]
PAUSES = set(PUNCT.values())


# --- Numbers in words -------------------------------------------------------------------------------------------------------
def read_digits(s: str) -> str:
    return "".join(DIGITS[int(d)] for d in s)


def _group(n: int, leading: bool) -> str:
    """0 < n < 10000 in words; `leading`: 10–19 as 十… (not 一十…)."""
    out, zero = "", False
    for unit, v in (("千", n // 1000), ("百", n // 100 % 10), ("十", n // 10 % 10), ("", n % 10)):
        if v == 0:
            zero = bool(out)
            continue
        if zero:
            out += "零"
            zero = False
        out += "十" if (unit == "十" and v == 1 and not out and leading) else DIGITS[v] + unit
    return out


def read_int(s: str) -> str:
    """An integer in words, grouped by 万 and 亿; nine digits or more, or a leading zero, digit by digit."""
    s = s.lstrip("+")
    if len(s) > 1 and s[0] == "0" or len(s) >= 9:
        return read_digits(s)
    n = int(s)
    if n == 0:
        return "零"
    parts, groups = [], []
    while n:
        groups.append(n % 10000)
        n //= 10000
    out = ""
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        if out and g < 1000:
            out += "零"
        out += _group(g, leading=not out) + ("", "万", "亿")[i]
    return out


def read_number(s: str) -> str:
    whole, _, frac = s.partition(".")
    neg = whole.startswith("-")
    whole = whole.lstrip("-")
    out = ("负" if neg else "") + (read_int(whole) if whole else "零")
    return out + ("点" + read_digits(frac) if frac else "")


NUM = r"\d+(?:\.\d+)?"
RULES = [
    # 1,280,000 → 1280000
    (re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])"), lambda m: m.group(0).replace(",", "")),
    # 2026/3/15, 2026-03-15 → 2026年3月15日
    (re.compile(r"(?<!\d)(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})(?!\d)"), lambda m: f"{m.group(1)}年{int(m.group(2))}月{int(m.group(3))}日"),
    # years: four digits before 年 digit by digit
    (re.compile(r"(?<!\d)(\d{4})(?=年)"), lambda m: read_digits(m.group(1))),
    # times: 8:30, 8:30:15
    (re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)"),
     lambda m: read_int(m.group(1)) + "点" + ("" if m.group(2) == "00" else ("零" + DIGITS[int(m.group(2)[1])] if m.group(2)[0] == "0" else read_int(m.group(2))) + "分")
     + (read_int(m.group(3)) + "秒" if m.group(3) else "")),
    # percent and per mille
    (re.compile(rf"(-?{NUM})\s*[%％]"), lambda m: "百分之" + read_number(m.group(1))),
    (re.compile(rf"(-?{NUM})\s*‰"), lambda m: "千分之" + read_number(m.group(1))),
    # fractions 1/2
    (re.compile(r"(?<![\d/])(\d{1,4})/(\d{1,4})(?![\d/])"), lambda m: read_int(m.group(2)) + "分之" + read_int(m.group(1))),
    # ranges 3-5, 3~5
    (re.compile(rf"(?<![\d.])({NUM})\s*[-~～–—]\s*({NUM})(?![\d.])"), lambda m: read_number(m.group(1)) + "到" + read_number(m.group(2))),
    # negatives
    # (after Chinese too: 降到了-5)
    (re.compile(rf"(?<![A-Za-z0-9.])-({NUM})"), lambda m: "负" + read_number(m.group(1))),
    # 2 before a measure word → 两
    (re.compile(r"(?<![\d.])2(?=[" + "".join(sorted(MEASURE)) + "])"), lambda m: "两"),
    # every other number
    (re.compile(NUM), lambda m: read_number(m.group(0))),
    # symbols
    # NFKC has made ℃ into °C
    (re.compile(r"°\s*C(?![A-Za-z])"), lambda m: "摄氏度"),
    (re.compile(r"°\s*F(?![A-Za-z])"), lambda m: "华氏度"),
    (re.compile(r"°"), lambda m: "度"),
]


def _sub(text, src, pattern, fn):
    """Applies one rule, keeping for every output character the index of the source character it came from (a replacement
    maps all its characters to the first character it replaced)."""
    out, out_src, pos = [], [], 0
    for m in pattern.finditer(text):
        out.append(text[pos:m.start()]); out_src.extend(src[pos:m.start()])
        r = fn(m)
        out.append(r); out_src.extend([src[m.start()]] * len(r))
        pos = m.end()
    out.append(text[pos:]); out_src.extend(src[pos:])
    return "".join(out), out_src


class Frontend:
    def __init__(self, data: Path = DATA):
        cfg = json.loads((data / "phonemes.json").read_text())
        self.ids = cfg["phoneme_id_map"]
        self.pad, self.bos, self.eos = self.ids["_"][0], self.ids["^"][0], self.ids["$"][0]
        self.phrases, self.chars = {}, {}
        for path, table in ((data / "phrases.tsv", self.phrases), (data / "chars.tsv", self.chars)):
            for line in path.read_text().splitlines():
                k, v = line.split("\t")
                table[k] = [s.split(",") if s != "-" else None for s in v.split(" ")]
        self.prefixes = {k[:i] for k in self.phrases for i in range(1, len(k) + 1)}
        self.t2s = {}
        for line in (data / "t2s.tsv").read_text().splitlines():
            k, v = line.split("\t")
            self.t2s[k] = v
        self.t2s_longest = max(map(len, self.t2s))
        self.english = {}
        for line in (data / "english.tsv").read_text().splitlines():
            k, v = line.split("\t")
            self.english[k] = v.split(" ")

    # 1. text clean-up
    def normalize(self, text: str):
        out, src = [], []
        for i, ch in enumerate(text):
            n = unicodedata.normalize("NFKC", ch)
            out.append(n); src.extend([i] * len(n))
        text = "".join(out)
        # Traditional → Simplified, longest match, only replacements of the same length (indices stay put)
        chars, i = list(text), 0
        while i < len(text):
            for L in range(min(self.t2s_longest, len(text) - i), 0, -1):
                v = self.t2s.get(text[i:i + L])
                if v is not None:
                    chars[i:i + L] = list(v)
                    i += L
                    break
            else:
                i += 1
        text = "".join(chars)
        for pattern, fn in RULES:
            text, src = _sub(text, src, pattern, fn)
        return text, src

    # 2. pypinyin's segmentation (pypinyin.seg.mmseg.Seg.cut with no_non_phrases=True)
    def cut(self, text: str):
        remain = text
        while remain:
            last_valid, last_index = "", 0
            for index in range(len(remain)):
                word = remain[:index + 1]
                if word in self.prefixes:
                    if word in self.phrases:
                        last_valid, last_index = word, index + 1
                else:
                    if last_valid:
                        yield last_valid
                        remain = remain[last_index:]
                    else:
                        yield remain[0]
                        remain = remain[1:]
                    break
            else:
                if last_valid:
                    yield last_valid
                    remain = remain[last_index:]
                else:
                    if remain not in self.phrases:
                        yield from remain
                    else:
                        yield remain
                    break

    def han(self, run: str):
        """A run of Chinese characters → one [initial, final, tone] (or None) per character."""
        items, prev = [], ""
        for word in self.cut(run):
            if word in self.phrases:
                sy = [list(s) if s else None for s in self.phrases[word]]
            else:
                sy = [list(self.chars[c][0]) if self.chars.get(c) and self.chars[c][0] else None for c in word]
            if word == "地" and prev and (prev[-1] in DE_AFTER or (len(prev) >= 2 and prev[-1] == prev[-2])):
                sy = [["d", "e", "5"]]
            for c, s in zip(word, sy):
                items.append((c, ALWAYS.get(c, s)))
            prev = word
        # 一 / 不 tone changes (Suxin's), but not in numbers or ordinals
        for i, (c, s) in enumerate(items):
            if not s or c not in "一不":
                continue
            nxt = items[i + 1][1] if i + 1 < len(items) else None
            if not nxt:
                continue
            before = items[i - 1][0] if i else ""
            after = items[i + 1][0]
            if c == "一":
                if before in NUMERAL or after in NUMERAL or after in "月日号":
                    continue
                s[2] = "2" if nxt[2] == "4" else ("4" if nxt[2] in "123" else s[2])
            elif nxt[2] == "4":
                s[2] = "2"
        return [s for _, s in items]

    # 3–4. symbols and ids
    def encode(self, text: str):
        """text → (ids, char) where char[i] is the index in `text` of the character id i came from (-1 for BOS / EOS)."""
        norm, src = self.normalize(text)
        syms = []  # (symbol, source index)
        i = 0
        while i < len(norm):
            c = norm[i]
            if HAN.match(c):
                j = i
                while j < len(norm) and HAN.match(norm[j]):
                    j += 1
                for k, s in enumerate(self.han(norm[i:j])):
                    if s:
                        syms.extend((x, src[i + k]) for x in s)
                i = j
            elif c.isascii() and c.isalpha():
                j = i
                while j < len(norm) and (norm[j].isascii() and norm[j].isalpha() or norm[j] == "'" and j + 1 < len(norm) and norm[j + 1].isascii() and norm[j + 1].isalpha()):
                    j += 1
                word = norm[i:j]
                ph = self.english.get(word.lower())
                if ph is None:
                    ph = [p for ch in word if ch.isalpha() for p in self.english["#" + ch.upper()]]
                syms.extend((p, src[i]) for p in ph)
                i = j
            elif c.isspace():
                j = i
                while j < len(norm) and norm[j].isspace():
                    j += 1
                if syms and syms[-1][0] not in GROUP_END:
                    syms.append((" ", src[i]))
                i = j
            elif c.isdigit():  # left over by the rules: digit by digit
                value = unicodedata.digit(c, None)
                if value is not None:
                    syms.extend((x, src[i]) for x in syllable(*LEFTOVER_DIGIT[value]))
                i += 1
            else:
                p = PUNCT.get(c)
                if p:
                    if syms and syms[-1][0] == " ":
                        syms.pop()
                    # A run of marks (……, ?!, NFKC's ......) is one pause: the strongest of them.
                    if syms and syms[-1][0] in PAUSES:
                        if p != "，" and syms[-1][0] == "，":
                            syms[-1] = (p, syms[-1][1])
                    else:
                        syms.append((p, src[i]))
                i += 1
        while syms and syms[-1][0] == " ":
            syms.pop()
        ids, char = [self.bos], [-1]
        for sym, at in syms:
            if sym not in self.ids:
                continue
            ids.extend(self.ids[sym]); char.extend([at] * len(self.ids[sym]))
            if sym in GROUP_END:
                ids.append(self.pad); char.append(at)
        ids.append(self.eos); char.append(-1)
        return ids, char


# --- Data --------------------------------------------------------------------------------------------------------------------
def build():
    from pypinyin import Style, pinyin
    from pypinyin.constants import PHRASES_DICT, PINYIN_DICT

    DATA.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    (DATA / "phonemes.json").write_text(json.dumps({"phoneme_id_map": cfg["phoneme_id_map"]}, ensure_ascii=False))

    def readings(word):
        ini = pinyin(word, style=Style.INITIALS, strict=False)
        fin = pinyin(word, style=Style.FINALS_TONE3, strict=False)
        return [syllable(i[0], f[0]) for i, f in zip(ini, fin)]

    def fmt(sy):
        return " ".join(",".join(s) if s else "-" for s in sy)

    with open(DATA / "phrases.tsv", "w") as f:
        for word in sorted(PHRASES_DICT):
            sy = PHRASE_ADD.get(word) or readings(word)
            if len(sy) == len(word):
                f.write(f"{word}\t{fmt(sy)}\n")
        for word, sy in PHRASE_ADD.items():
            if word not in PHRASES_DICT:
                f.write(f"{word}\t{fmt(sy)}\n")
    with open(DATA / "chars.tsv", "w") as f:
        for code in sorted(PINYIN_DICT):
            c = chr(code)
            sy = [CHAR_FIX[c]] if c in CHAR_FIX else readings(c)
            if len(sy) == 1 and sy[0]:
                f.write(f"{c}\t{fmt(sy)}\n")

    import opencc
    d = Path(opencc.__file__).parent / "dictionary"
    t2s = {}
    for name in ("TSCharacters.txt", "TSPhrases.txt"):
        for line in (d / name).read_text().splitlines():
            if "\t" not in line:
                continue
            k, v = line.split("\t")
            v = v.split(" ")[0]
            # same length (indices stay put), and nothing outside the Basic Multilingual Plane the reading can't use
            if len(k) == len(v) and k != v and all(ord(c) <= 0xFFFF for c in k + v):
                t2s[k] = v
    with open(DATA / "t2s.tsv", "w") as f:
        for k in sorted(t2s):
            f.write(f"{k}\t{t2s[k]}\n")

    import piper
    import wordfreq
    from piper.phonemize_espeak import EspeakPhonemizer
    espeak = EspeakPhonemizer(str(Path(piper.__file__).parent / "espeak-ng-data"))

    def phonemes(word):
        return [p for sent in espeak.phonemize("en-us", word) for p in sent if p.strip()]

    # espeak's phonemes (kept for english.py, not packed), then the table of Mandarin syllables the pack carries
    with open(DATA / "english_ipa.tsv", "w") as f:
        for L in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            f.write(f"#{L}\t{' '.join(phonemes(L))}\n")
        for w in wordfreq.top_n_list("en", 100000):
            if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", w) and len(w) > 1:
                ph = phonemes(w)
                if ph:
                    f.write(f"{w}\t{' '.join(ph)}\n")
    import english
    table = english.table()
    with open(DATA / "english.tsv", "w") as f:
        for k in sorted(table):
            f.write(f"{k}\t{' '.join(x for syl in table[k] for x in syl)}\n")
    for p in sorted(DATA.iterdir()):
        print(f"{p.name}: {p.stat().st_size:,} B")
    print("english words:", len(table))


def golden(path):
    f = Frontend()
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            ids, char = f.encode(line)
            out.append({"text": line, "ids": ids, "chars": char})
    Path(str(path) + ".golden.json").write_text(json.dumps(out, ensure_ascii=False))
    print(len(out), "lines")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "say":
        f = Frontend()
        text = " ".join(sys.argv[2:])
        print(f.normalize(text)[0])
        ids, char = f.encode(text)
        inv = {v[0]: k for k, v in f.ids.items()}
        print(" ".join(inv[i] for i in ids))
        print(ids)
    elif cmd == "golden":
        golden(sys.argv[2])
    else:
        sys.exit(__doc__)
