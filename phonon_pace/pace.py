"""Words a second of every Phonon voice on one four-sentence passage of its language: the numbers behind PhononPace in both apps.

    hf download gradium/phonon-40m-multilingual --local-dir phonon40m   (and phonon-voices, see android/phonon_pack.py)
    PTTS=/path/to/ptts python3 phonon_pace/pace.py

factor = clamp(2.8 / words a second, 0.9, 1.1). Writes pace.json (rendered wavs go to wav/, ignored).
"""
import json, subprocess, unicodedata, pathlib, wave
def fold(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
HERE=pathlib.Path(__file__).resolve().parent
M=HERE.parent/'phonon40m'
import os
P=os.environ.get('PTTS','ptts')  # the ptts CLI (cargo build --release -p ptts --features cli --bin ptts in xn-ptts)
T=HERE; O=HERE/'wav'; O.mkdir(exist_ok=True)
res={}
for l in open(M/'phonon-voices/voices.jsonl'):
    r=json.loads(l)
    if r['locale']=='characters': continue
    lang=r['language']; k=fold(r['name']); text=(T/f'{lang}.txt').read_text().strip()
    wav=O/f'{lang}-{k}.wav'
    if not wav.exists():
        subprocess.run([P,'--model',str(M),'--weights','model.q8.gguf','--quant','q8','--voice-catalog',str(M/'phonon-voices'),'--voice',r['name'],'--lang',lang,'-o',str(wav),text],check=True,capture_output=True)
    w=wave.open(str(wav)); d=w.getnframes()/w.getframerate(); w.close()
    res[f'{lang}.{k}']=round(len(text.split())/d,3)
json.dump(res,open(HERE/'pace.json','w'),indent=1)
import statistics
for lang in ['en','fr','de','es','pt']:
    v=[x for kk,x in res.items() if kk.startswith(lang+'.')]
    print(lang,len(v),'min',min(v),'median',round(statistics.median(v),2),'max',max(v))
