"""The voice picker's ▶︎ preview clips of the 90 Phonon voices, `<lang>_<Key>.m4a`, the same files in both apps
(bookmuse_ios GMReader/Resources/VoiceClips, bookmuse_android core/ondevice/src/main/assets/voiceclips).

    PTTS=/path/to/ptts python3 phonon_pace/clips.py     (after pace.py: each voice is said at its pace.json `delta`)

Each clip is one sentence rendered by the ptts CLI at the voice's `duration_delta`, trimmed and faded as the apps trim a unit
(40 ms of each end's silence kept, a 15 ms cosine fade), and encoded as AAC 64 kb/s. Writes dist/clips/phonon/.
"""
import array, json, math, os, pathlib, subprocess, unicodedata, wave

TEXTS = {
    'en': 'This is how I sound when I read your book aloud.',
    'fr': 'Voilà comment je sonne quand je lis votre livre à voix haute.',
    'de': 'So klinge ich, wenn ich dein Buch laut vorlese.',
    'es': 'Así suena mi voz cuando leo tu libro en voz alta.',
    'pt': 'É assim que eu soo quando leio seu livro em voz alta.',
}

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
M = ROOT / 'phonon40m'
P = os.environ.get('PTTS', 'ptts')
OUT = ROOT / 'dist/clips/phonon'
WORK = ROOT / 'work/phonon-clips'
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
pace = json.load(open(HERE / 'pace.json'))


def fold(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def trim_and_fade(x, rate):
    """`VieNeuAudio.trimAndFade`: edge silence is 10 ms windows under −40 dB."""
    win = rate // 100
    loud = [20 * math.log10(math.sqrt(sum(v * v for v in x[i:i + win]) / win) + 1e-9) > -40 for i in range(0, len(x) - win + 1, win)]
    if True not in loud:
        return []
    lead = loud.index(True) * win
    tail = (len(loud) - 1 - loud[::-1].index(True)) * win + win
    keep = int(0.04 * rate)
    out = x[max(0, lead - keep):min(len(x), tail + keep)]
    n = min(int(0.015 * rate), len(out) // 2)
    for i in range(n):
        ramp = 0.5 - 0.5 * math.cos(math.pi * i / max(n - 1, 1))
        out[i] *= ramp
        out[-1 - i] *= ramp
    return out


for line in open(M / 'phonon-voices/voices.jsonl'):
    r = json.loads(line)
    if r['locale'] == 'characters':
        continue
    lang, name, key = r['language'], r['name'], fold(r['name'])
    delta = pace[f'{lang}.{key}']['delta']
    made = WORK / f'{lang}_{key}.wav'
    subprocess.run([P, '--model', str(M), '--weights', 'model.safetensors', '--voice-catalog', str(M / 'phonon-voices'), '--voice', name,
                    '--lang', lang, '--condition', f'duration_delta={delta}', '-o', str(made), TEXTS[lang]], check=True, capture_output=True)
    with wave.open(str(made)) as w:
        rate, raw = w.getframerate(), w.readframes(w.getnframes())
    samples = [v / 32768 for v in array.array('h', raw)]
    trimmed = trim_and_fade(samples, rate)
    pcm = WORK / f'{lang}_{key}.trim.wav'
    with wave.open(str(pcm), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(array.array('h', [max(-32767, min(32767, int(v * 32767))) for v in trimmed]).tobytes())
    subprocess.run(['afconvert', '-f', 'm4af', '-d', 'aac', '-b', '64000', str(pcm), str(OUT / f'{lang}_{key}.m4a')], check=True)
    print(f'{lang}_{key}  delta {delta}  {len(trimmed) / rate:.2f} s')
