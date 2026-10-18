"""UNSW->Bot-IoT stage-1 attack recall on one million genuine attack rows."""
import os
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score,recall_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from run_experiment import EdgeGuard,write_json


def main():
    root=Path('/workspace/edgeguard');out=root/'results/botiot_transfer';out.mkdir(parents=True,exist_ok=True)
    features=['sbytes','dbytes','spkts','dpkts','rate']
    pre=joblib.load(root/'results/hybrid/preprocessing.joblib');split=np.load(root/'results/hybrid/split.npz')
    raw=pd.read_csv(root/'data/UNSW_NB15_testing-set.csv')
    # Split/deduplicate using the original 39 predictors, NOT the restricted five.
    data=raw[pre['numeric_columns']+['attack_cat']].drop_duplicates().reset_index(drop=True)
    X=data[features].replace([np.inf,-np.inf],np.nan).to_numpy()
    y=pre['encoder'].transform(data.attack_cat);normal=int(pre['encoder'].transform(['Normal'])[0])
    tr,te=split['train'],split['test'];imputer=SimpleImputer(strategy='median').fit(X[tr])
    scaler=StandardScaler().fit(imputer.transform(X[tr]));Xtr=scaler.transform(imputer.transform(X[tr]));Xte=scaler.transform(imputer.transform(X[te]))
    with threadpool_limits(limits=1):
        model=EdgeGuard(normal).fit(Xtr,y[tr]);binary=model.stage1_.predict(Xte)
        results=dict(features=features,in_domain_stage1_f1=f1_score(y[te]!=normal,binary),
            in_domain_stage1_attack_recall=recall_score(y[te]!=normal,binary),
            in_domain_macro_f1=f1_score(y[te],model.predict(Xte),average='macro'),
            target_source='Dataset/reduced_data_1.csv from public 5% release mirror',
            caveat='Attack-only target: reports attack recall, not multiclass macro F1. A fresh five-feature model is fit on original UNSW training rows. No target recalibration or retraining.')
        count=flagged=0;categories={};ranges=[]
        with zipfile.ZipFile(root/'data/Bot-IoT-mirror.zip') as z:
            with z.open('Dataset/reduced_data_1.csv') as f:
                for chunk in pd.read_csv(f,usecols=features+['attack','category'],chunksize=100000):
                    assert (chunk.attack==1).all(),'Selected transfer file is not attack-only'
                    Xtarget=chunk[features].replace([np.inf,-np.inf],np.nan).to_numpy()
                    pred=model.stage1_.predict(scaler.transform(imputer.transform(Xtarget)))
                    count+=len(chunk);flagged+=int(pred.sum());ranges.append(Xtarget)
                    for k,v in chunk.category.value_counts().items():categories[k]=categories.get(k,0)+int(v)
        assert count==1000000,count
        target=np.concatenate(ranges)
        results.update(target_flows=count,flagged_attacks=flagged,stage1_attack_recall=flagged/count,
            categories=categories,
            unsw_training_medians=dict(zip(features,np.nanmedian(X[tr],axis=0))),
            botiot_medians=dict(zip(features,np.nanmedian(target,axis=0))))
        write_json(out/'results.json',results)
        joblib.dump(dict(model=model,imputer=imputer,scaler=scaler,features=features),out/'five_feature_model.joblib')
        print(results,flush=True);print('TRANSFER COMPLETE',flush=True)


if __name__=='__main__':main()
