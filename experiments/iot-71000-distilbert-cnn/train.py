"""Conditional rerun of a disclosed architecture hypothesis on supplied REAL rows.

No synthetic fallback and no silent relabeling. Defaults require manuscript labels
and public pretrained weights; neither is claimed available by the audit report.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset
from model import DualHead,multitask_loss,predict_routed

PAPER_CLASSES=['ArpSpoof','Benign','Control&Freeze','DriftOff','GeneratorControl','Industroyer']


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',default='proposed',choices=['proposed','distilbert_only','cnn_only','without_attention','focal'])
    p.add_argument('--initialization',choices=['pretrained','random'],default='pretrained')
    p.add_argument('--checkpoint',default='distilbert/distilbert-base-uncased')
    p.add_argument('--tokens',type=int,default=1);p.add_argument('--lambda-binary',type=float,default=.5)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--allow-canonical-five-class-diagnostic',action='store_true');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not 0<=a.lambda_binary<=1:raise ValueError('lambda-binary must be in [0,1]')
    if a.epochs<1:raise ValueError('At least one epoch is required.')
    if not a.csv.is_file():raise FileNotFoundError('Real feature table missing; run fetch_data.py after network access is applied. No synthetic substitution is permitted.')
    frame=pd.read_csv(a.csv,low_memory=False)
    if 'label' not in frame:raise ValueError('Feature table must contain a documented categorical label column.')
    classes=sorted(frame.label.unique());canonical=[c for c in PAPER_CLASSES if c!='GeneratorControl']
    if classes!=PAPER_CLASSES:
        if not (a.allow_canonical_five_class_diagnostic and classes==canonical):
            raise ValueError(f'Found {classes}; manuscript requires {PAPER_CLASSES}. Obtain author labeling rules; do not invent GeneratorControl attacks. Explicit canonical diagnostic flag permits a DIFFERENT five-class task.')
    benign=classes.index('Benign');attack_ids=[i for i in range(len(classes)) if i!=benign]
    excluded={'label','timestamp','malicious','attack','attack_type','binary_label','event_id','event','id','target'}
    columns=[c for c in frame.select_dtypes(include=np.number).columns if c.lower() not in excluded]
    if not columns:raise ValueError('No numeric predictors.')
    y=pd.Categorical(frame.label,categories=classes).codes
    ids=np.arange(len(frame));train,remainder=train_test_split(ids,test_size=.4,stratify=y,random_state=a.seed)
    val,test=train_test_split(remainder,test_size=.5,stratify=y[remainder],random_state=a.seed)
    np.savez_compressed(a.output/'split.npz',train=train,validation=val,test=test)
    raw=frame[columns].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
    keep=np.nanstd(raw[train],axis=0)>0;columns=np.array(columns)[keep].tolist();raw=raw[:,keep]
    imputer=SimpleImputer(strategy='median').fit(raw[train]);scaler=StandardScaler().fit(imputer.transform(raw[train]))
    x=scaler.transform(imputer.transform(raw)).astype('float32')
    joblib.dump(dict(columns=columns,classes=classes,imputer=imputer,scaler=scaler),a.output/'preprocessing.joblib')
    attack_map={i:j for j,i in enumerate(attack_ids)};yb=(y!=benign).astype('int64');ya=np.array([attack_map.get(i,-1) for i in y])
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    model=DualHead(len(columns),len(attack_ids),a.variant,a.tokens,a.initialization,a.checkpoint)
    count=np.bincount(ya[train][yb[train].astype(bool)],minlength=len(attack_ids))
    if (count==0).any():raise ValueError('Some attack classes have zero training rows.')
    weights=torch.tensor(count.sum()/(len(count)*count),dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
    generator=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(yb[train]),torch.from_numpy(ya[train])),batch_size=64,shuffle=True,generator=generator)
    history=[];best=-np.inf;stale=0;selected_epoch=None
    for epoch in range(1,a.epochs+1):
        model.train();loss_total=0;begin=time.perf_counter()
        for batch,binary,family in loader:
            optimizer.zero_grad(set_to_none=True);b,m=model(batch)
            loss=multitask_loss(b,m,binary,family,weights,a.lambda_binary,a.variant=='focal')
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            loss.backward();optimizer.step();loss_total+=float(loss.detach())*len(batch)
        model.eval();vp=[]
        with torch.inference_mode():
            for start in range(0,len(val),64):vp.extend(predict_routed(model,torch.from_numpy(x[val[start:start+64]]),benign,attack_ids)[0].tolist())
        score=f1_score(y[val],vp,labels=np.arange(len(classes)),average='macro',zero_division=0)
        record=dict(epoch=epoch,training_loss=loss_total/len(train),validation_macro_f1=score,seconds=time.perf_counter()-begin)
        history.append(record);pd.DataFrame(history).to_csv(a.output/'history.csv',index=False);print(json.dumps(record),flush=True)
        if score>best:
            best=score;stale=0;selected_epoch=epoch;torch.save(model.state_dict(),a.output/'best_model.pt')
        else:stale+=1
        if stale>=5:break
    model.load_state_dict(torch.load(a.output/'best_model.pt',weights_only=True));model.eval();pred=[];probs=[]
    with torch.inference_mode():
        for start in range(0,len(test),64):
            pr,po=predict_routed(model,torch.from_numpy(x[test[start:start+64]]),benign,attack_ids)
            pred.extend(pr.tolist());probs.extend(po.tolist())
    pred=np.array(pred);probs=np.array(probs)
    np.savez_compressed(a.output/'test_predictions.npz',indices=test,true=y[test],pred=pred,probabilities=probs)
    result=dict(status='independent_assumed_implementation_not_author_code',variant=a.variant,initialization=a.initialization,
        checkpoint=a.checkpoint,tokens=a.tokens,lambda_binary=a.lambda_binary,classes=classes,
        source_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest(),seed=a.seed,selected_epoch=selected_epoch,
        train_rows=len(train),validation_rows=len(val),test_rows=len(test),
        accuracy=accuracy_score(y[test],pred),macro_f1=f1_score(y[test],pred,average='macro',zero_division=0),
        weighted_f1=f1_score(y[test],pred,average='weighted',zero_division=0),
        macro_ovr_auc=roc_auc_score(y[test],probs,multi_class='ovr',average='macro'),
        confusion_matrix=confusion_matrix(y[test],pred).tolist(),
        per_class=classification_report(y[test],pred,target_names=classes,output_dict=True,zero_division=0),
        caveat='Random row stratification is described by the manuscript; it does not ensure event/temporal independence. Transformer init, tokens, branch/fusion definitions, loss masking, lambda, seed and checkpoint are explicit assumptions.')
    (a.output/'results.json').write_text(json.dumps(result,indent=2));print('TRAINING RUN COMPLETE',flush=True)


if __name__=='__main__':main()
