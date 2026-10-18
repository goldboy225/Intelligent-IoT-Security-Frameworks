"""Build a transparent alternative sample from the public Bot-IoT 5% release."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES=['pkts','bytes','dur','mean','stddev','sum','min','max','spkts','dpkts','sbytes','dbytes','rate','srate','drate']


def main():
    root=Path('/workspace/edgeguard/data');archive=root/'Bot-IoT-mirror.zip'
    rng=np.random.default_rng(42);parts=[];manifest=[]
    with zipfile.ZipFile(archive) as z:
        for filename in sorted(n for n in z.namelist() if n.endswith('.csv')):
            counts={};rare=[];retained={}
            with z.open(filename) as f:
                for chunk in pd.read_csv(f,usecols=FEATURES+['attack','category','pkSeqID'],chunksize=100000):
                    for name,group in chunk.groupby('category'):
                        counts[name]=counts.get(name,0)+len(group)
                        if name in ['Normal','Theft']:rare.append(group)
                        else:
                            group=group.assign(sample_key=rng.random(len(group)))
                            prior=retained.get(name)
                            retained[name]=pd.concat([prior,group],ignore_index=True).nsmallest(3000,'sample_key') if prior is not None else group.nsmallest(3000,'sample_key')
            selected=pd.concat(rare+list(retained.values()),ignore_index=True).drop(columns='sample_key',errors='ignore')
            selected['source_file']=filename;parts.append(selected)
            manifest.append(dict(file=filename,raw_counts=counts,selected_counts=selected.category.value_counts().to_dict()))
    result=pd.concat(parts,ignore_index=True)
    result.to_csv(root/'Bot-IoT-5percent-independent-sample.csv',index=False)
    metadata=dict(source='https://github.com/AG24-hub/Gradient-Boosting-algorithm-on-BOT-IOT-Dataset/blob/main/Dataset.zip',
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=manifest,
        sample_seed=42,rows=len(result),features=FEATURES,
        limitation='Alternative four-file 5% release sample, NOT paper Table 5 original 40-scenario-file sample. All Normal/Theft plus up to 3000 random rows per other category per file. No claim of exact Bot-IoT same-dataset reproduction.')
    (root/'botiot_sample_manifest.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__': main()
