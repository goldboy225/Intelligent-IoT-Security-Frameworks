"""Preserve the specified baseline; diagnose L2=0 instability separately."""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from run_experiment import balanced_data, hist, evaluate, write_json


def main():
    root=Path('/workspace/edgeguard/results/hybrid')
    out=Path('/workspace/edgeguard/results/histgb_diagnostic');out.mkdir(exist_ok=True)
    for d in ['models','metrics','per_class','predictions']:(out/d).mkdir(exist_ok=True)
    pre=joblib.load(root/'preprocessing.joblib');split=np.load(root/'split.npz')
    raw=pd.read_csv('/workspace/edgeguard/data/UNSW_NB15_testing-set.csv')
    data=raw[pre['numeric_columns']+['attack_cat']].drop_duplicates().reset_index(drop=True)
    X=pre['scaler'].transform(pre['imputer'].transform(data[pre['numeric_columns']].to_numpy()))[:,pre['selected_indices']]
    y=pre['encoder'].transform(data.attack_cat)
    tr,te=split['train'],split['test'];Xtr,Xte=X[tr],X[te];ytr,yte=y[tr],y[te]
    baseline=joblib.load(root/'models/HistGBM.joblib')
    trajectory=[]
    with threadpool_limits(limits=1):
        margins=baseline._raw_predict(Xte)
        for iteration,probs in enumerate(baseline.staged_predict_proba(Xte),1):
            if iteration in [1,10,25,50,100,150,200]:
                trajectory.append(dict(iteration=iteration,macro_f1=f1_score(yte,baseline.classes_[probs.argmax(axis=1)],average='macro')))
        pd.DataFrame(trajectory).to_csv(out/'prespecified_L2_0_trajectory.csv',index=False)
        write_json(out/'diagnosis.json',dict(raw_margin_min=float(margins.min()),raw_margin_max=float(margins.max()),
            interpretation='L2=0 model develops extreme raw margins; held-out F1 deteriorates after early iterations. No iteration is selected from held-out scores. L2=1 is a separate sensitivity analysis, not a replacement baseline.'))
        Xb,yb,w=balanced_data(Xtr,ytr,True,True)
        model=hist(l2=1.).fit(Xb,yb,sample_weight=w)
        record,rows=evaluate('HistGBM_L2_1_sensitivity',model,Xte,yte,pre['encoder'],
            np.argsort(np.bincount(ytr))[:3],out,Xtr.min(axis=0),Xtr.max(axis=0))
        pd.DataFrame([record]).to_csv(out/'summary.csv',index=False)
        pd.DataFrame(rows).to_csv(out/'corruptions.csv',index=False)
    print('DIAGNOSTIC COMPLETE',flush=True)


if __name__=='__main__':main()
