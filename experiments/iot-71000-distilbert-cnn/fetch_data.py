"""Fetch the public IPAL Sherlock mirror; no substitute data on failure."""
import gzip
import hashlib
import json
from pathlib import Path
import requests

COMMIT='2fae8546757f82b4840a492cf6351d28f3727e60'
SHA='38b2f84c21cb2ca92aa4146f9defde74a1e5c0207b616b3d8ac1faac19819e08'
ROOT=Path('/workspace/sherlock-review/data')
MAPPING={'arp-spoof':'ArpSpoof','arp-spoof-dos':'ArpSpoof',
    'control-and-freeze':'Control&Freeze','drift-off':'DriftOff','industroyer':'Industroyer'}


def canonical_label(marker,events):
    if marker is False or (isinstance(marker,str) and '(benign event)' in marker):return 'Benign'
    event=events.get(str(marker))
    if event is None:raise ValueError(f'Unknown event marker: {marker!r}')
    kind=event['description'].split(':')[0].lower()
    if kind=='control center':return 'Benign'
    if kind not in MAPPING:raise ValueError(f'Unmapped attack type: {kind!r}')
    return MAPPING[kind]


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    state=ROOT/'test.n302.state.gz';catalog=ROOT/'test.events.json'
    if not state.exists():
        url=f'https://media.githubusercontent.com/media/ipal-ids/ipal_datasets/{COMMIT}/Sherlock/ipal/01-Basic/test.n302.state.gz'
        response=requests.get(url,timeout=120);response.raise_for_status();content=response.content
        assert hashlib.sha256(content).hexdigest()==SHA,'State checksum mismatch; refusing content'
        state.write_bytes(content)
    assert hashlib.sha256(state.read_bytes()).hexdigest()==SHA,'Local checksum mismatch'
    if not catalog.exists():
        url=f'https://raw.githubusercontent.com/ipal-ids/ipal_datasets/{COMMIT}/Sherlock/ipal/01-Basic/test.events.json'
        response=requests.get(url,timeout=30);response.raise_for_status();json.loads(response.text)
        catalog.write_bytes(response.content)
    events={str(e['id']):e for e in json.loads(catalog.read_text())}
    rows=[]
    with gzip.open(state,'rt') as stream:
        for line in stream:
            record=json.loads(line);features={k:float(v) for k,v in record['state'].items() if isinstance(v,(int,float,bool))}
            rows.append(dict(**features,label=canonical_label(record['malicious'],events),timestamp=record['timestamp']))
    import pandas as pd
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'canonical_process_state.csv',index=False)
    summary=dict(rows=len(frame),class_counts=frame.label.value_counts().to_dict(),
        numeric_state_features=len(frame.columns)-2,state_sha256=SHA,source_commit=COMMIT,
        event_catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        label_policy='Original benign maintenance stays benign; no GeneratorControl attack fabricated.',
        caveat='Process-state view only, not an author-specified network/process fusion table. Five canonical classes are not the manuscript six-class task.')
    (ROOT/'data_manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
