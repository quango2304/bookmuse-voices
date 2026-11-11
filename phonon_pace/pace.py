"""Words a second of every Phonon voice on one four-sentence passage of its language: the rule behind PhononPace in both apps.

    hf download gradium/phonon-40m-multilingual --local-dir phonon40m   (and phonon-voices, see android/phonon_pack.py)
    PTTS=/path/to/ptts python3 phonon_pace/pace.py

Each voice is rendered at its own pace (`duration_delta` 0) and at the value the apps give it (listening.md, Pace): SLOWER for a voice
faster than OWN_PACE_MAX words a second, 0 for the others. Writes pace.json, {voice: {"own": w/s, "delta": value, "paced": w/s}}, and
prints the voices that keep their own pace (the list in Android's PhononPace.kt and ios/phonon_export.sh). Rendered wavs go to wav/
(ignored).
"""
import json, os, pathlib, statistics, subprocess, unicodedata, wave

SLOWER = 0.15
OWN_PACE_MAX = 3.1


def fold(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


HERE = pathlib.Path(__file__).resolve().parent
M = HERE.parent / 'phonon40m'
P = os.environ.get('PTTS', 'ptts')  # the ptts CLI (cargo build --release -p ptts --features cli --bin ptts in xn-ptts)
O = HERE / 'wav'
O.mkdir(exist_ok=True)


def words_a_second(lang, name, text, delta):
    wav = O / f'{lang}-{fold(name)}-{delta}.wav'
    if not wav.exists():
        subprocess.run([P, '--model', str(M), '--weights', 'model.q8.gguf', '--quant', 'q8', '--voice-catalog', str(M / 'phonon-voices'),
                        '--voice', name, '--lang', lang, '--condition', f'duration_delta={delta}', '-o', str(wav), text],
                       check=True, capture_output=True)
    with wave.open(str(wav)) as w:
        seconds = w.getnframes() / w.getframerate()
    return round(len(text.split()) / seconds, 3)


res = {}
for line in open(M / 'phonon-voices/voices.jsonl'):
    r = json.loads(line)
    if r['locale'] == 'characters':
        continue
    lang, name = r['language'], r['name']
    text = (HERE / f'{lang}.txt').read_text().strip()
    own = words_a_second(lang, name, text, 0)
    delta = SLOWER if own > OWN_PACE_MAX else 0
    paced = own if delta == 0 else words_a_second(lang, name, text, delta)
    res[f'{lang}.{fold(name)}'] = {'own': own, 'delta': delta, 'paced': paced}
json.dump(res, open(HERE / 'pace.json', 'w'), indent=1)

for lang in ['en', 'fr', 'de', 'es', 'pt']:
    v = [x for k, x in res.items() if k.startswith(lang + '.')]
    own, paced = [x['own'] for x in v], [x['paced'] for x in v]
    print(f'{lang} {len(v)}  own {min(own)}–{max(own)} (median {statistics.median(own):.2f})'
          f'  paced {min(paced)}–{max(paced)} (median {statistics.median(paced):.2f})')
print('own pace:', ' '.join(k.split('.', 1)[1] for k, x in res.items() if x['delta'] == 0))
