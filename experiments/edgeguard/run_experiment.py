"""Independent implementation of EdgeGuard-IoT; see README for assumptions.

Real UNSW-NB15 data only. Never substitutes synthetic data for benchmark results.
"""
import argparse
import hashlib
import inspect
import json
import os
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, radius_neighbors_graph
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

SEED = 42
EXPECTED = {'Normal':34204,'Exploits':7548,'Fuzzers':4781,'Generic':3655,
    'Reconnaissance':2664,'DoS':1657,'Analysis':394,'Shellcode':378,'Backdoor':320,'Worms':44}
PAPER_FEATURES = ['sbytes','dbytes','smean','dmean','spkts','dpkts','sloss',
    'response_body_len','rate','sload','dload','tcprtt','synack','sttl',
    'ct_srv_dst','ct_srv_src','ct_dst_sport_ltm','ct_dst_src_ltm','ct_src_ltm','ct_src_dport_ltm']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x:
        x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)))


def augment(X, y, weights, seed):
    """One clean, one Gaussian, one dropout view; scalar intensity per view."""
    rng = np.random.default_rng(seed)
    sigma, probability = rng.uniform(0, .3, size=2)
    noisy = X + rng.normal(0, sigma, X.shape)
    dropped = X * (rng.random(X.shape) >= probability)
    return (np.concatenate([X,noisy,dropped]), np.tile(y,3),
            None if weights is None else np.tile(weights,3))


def balanced_data(X, y, smote, weighting):
    if smote:
        labels, counts = np.unique(y, return_counts=True)
        targets = {int(k):4000 for k,n in zip(labels,counts) if n < 4000}
        if targets:
            X,y = SMOTE(sampling_strategy=targets,k_neighbors=5,random_state=SEED).fit_resample(X,y)
    return X,y,compute_sample_weight('balanced',y) if weighting else None


def hist(l2=1.):
    return HistGradientBoostingClassifier(max_iter=200,learning_rate=.12,
        max_leaf_nodes=31,l2_regularization=l2,early_stopping=False,random_state=SEED)


class EdgeGuard(ClassifierMixin, BaseEstimator):
    def __init__(self, normal_id, smote=True, weighting=True, augmentation=True):
        self.normal_id=normal_id
        self.smote=smote
        self.weighting=weighting
        self.augmentation=augmentation

    def fit(self, X, y):
        self.classes_=np.unique(y)
        self.n_features_in_=X.shape[1]
        binary=(y != self.normal_id).astype(int)
        X1,y1,w1=balanced_data(X,binary,False,self.weighting)
        mask=binary.astype(bool)
        X2,y2,w2=balanced_data(X[mask],y[mask],self.smote,self.weighting)
        self.stage2_counts_={int(k):int(v) for k,v in zip(*np.unique(y2,return_counts=True))}
        if self.augmentation:
            X1,y1,w1=augment(X1,y1,w1,SEED)
            X2,y2,w2=augment(X2,y2,w2,SEED+1)
        self.stage1_=hist().fit(X1,y1,sample_weight=w1)
        self.stage2_=hist().fit(X2,y2,sample_weight=w2)
        return self

    def predict(self, X):
        routed=self.stage1_.predict(X).astype(bool)
        result=np.full(len(X),self.normal_id,dtype=int)
        if routed.any(): result[routed]=self.stage2_.predict(X[routed])
        return result


def corrupt(X, family, level, low, high, seed):
    rng=np.random.default_rng(seed)
    if family=='noise': return X+rng.normal(0,level,X.shape)
    if family=='dropout': return X*(rng.random(X.shape)>=level)
    if family=='quantization':
        # Range comes from training rows only; clamp held-out outliers to that range.
        step=np.where(high>low,(high-low)/(int(level)-1),1.)
        return np.round((np.clip(X,low,high)-low)/step)*step+low
    raise ValueError(family)


def baseline_models():
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        'HistGBM':hist(l2=0.),
        'XGBoost':XGBClassifier(n_estimators=250,max_depth=8,learning_rate=.15,
            tree_method='hist',n_jobs=1,random_state=SEED),
        'LightGBM':LGBMClassifier(n_estimators=250,num_leaves=48,learning_rate=.10,
            n_jobs=1,random_state=SEED,verbosity=-1),
        'RandomForest':RandomForestClassifier(n_estimators=100,max_depth=25,n_jobs=1,random_state=SEED),
        'ExtraTrees':ExtraTreesClassifier(n_estimators=100,n_jobs=1,random_state=SEED),
        'DecisionTree':DecisionTreeClassifier(max_depth=18,random_state=SEED),
        'MLP':MLPClassifier(hidden_layer_sizes=(96,48),early_stopping=True,random_state=SEED),
        'kNN':KNeighborsClassifier(n_neighbors=7,n_jobs=1),
        'AdaBoost':AdaBoostClassifier(random_state=SEED),
        'LogReg':LogisticRegression(random_state=SEED),
        'LinearSVC':LinearSVC(random_state=SEED),
        'GaussianNB':GaussianNB(),
    }


def fit_weighted(model,X,y,weights):
    if 'sample_weight' in inspect.signature(model.fit).parameters:
        model.fit(X,y,sample_weight=weights)
    else: model.fit(X,y)
    return model


def scores(y,pred,classes,rare_ids):
    return dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=classes,average='macro',zero_division=0),
        weighted_f1=f1_score(y,pred,labels=classes,average='weighted',zero_division=0),
        mcc=matthews_corrcoef(y,pred),rare_class_recall=float(np.mean(
            recall_score(y,pred,labels=rare_ids,average=None,zero_division=0))))


def evaluate(name,model,X,y,encoder,rare_ids,out,low,high,robust=True):
    labels=np.arange(len(encoder.classes_))
    pred=model.predict(X)
    result=dict(model=name,**scores(y,pred,labels,rare_ids))
    report=classification_report(y,pred,labels=labels,target_names=encoder.classes_,output_dict=True,zero_division=0)
    write_json(out/'per_class'/f'{name}.json',report)
    np.save(out/'predictions'/f'{name}.npy',pred)
    # Warm full held-out batch, then five single-core passes. Prediction only, no preprocessing.
    model.predict(X)
    times=[]
    for _ in range(5):
        begin=time.perf_counter();model.predict(X);times.append(time.perf_counter()-begin)
    result['throughput_flows_s']=len(X)/np.median(times)
    result['batch_amortized_ms_flow']=1000*np.median(times)/len(X)
    result['inference_pass_seconds']=times
    target=out/'models'/f'{name}.joblib'
    joblib.dump(model,target,compress=0)
    result['serialized_bytes']=target.stat().st_size
    if isinstance(model,EdgeGuard):
        binary=(y!=model.normal_id).astype(int)
        routed=model.stage1_.predict(X)
        result.update(stage1_binary_f1=f1_score(binary,routed),
            stage1_attack_recall=recall_score(binary,routed),
            stage1_roc_auc=roc_auc_score(binary,model.stage1_.predict_proba(X)[:,1]),
            escalation_rate=float(routed.mean()),stage2_smote_counts=model.stage2_counts_)
    corruption_rows=[]
    if robust:
        for family,levels in [('noise',[.1,.2,.3,.5]),('dropout',[.1,.2,.3,.5]),('quantization',[16,8,4,2])]:
            for j,level in enumerate(levels):
                altered=corrupt(X,family,level,low,high,SEED+100+j)
                f1=f1_score(y,model.predict(altered),labels=labels,average='macro',zero_division=0)
                corruption_rows.append(dict(model=name,family=family,level=level,macro_f1=f1,
                    retention_ratio=f1/result['macro_f1'] if result['macro_f1'] else None))
        result['mean_retained_macro_f1']=float(np.mean([r['macro_f1'] for r in corruption_rows]))
        result['performance_retention_ratio']=result['mean_retained_macro_f1']/result['macro_f1']
    else: result['robustness_status']='not run: kNN excluded as in paper'
    write_json(out/'metrics'/f'{name}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ['model','accuracy','macro_f1','mean_retained_macro_f1']}),flush=True)
    return result,corruption_rows


def prepare(csv,out,feature_mode,dataset='unsw'):
    raw=pd.read_csv(csv,low_memory=False)
    target='Attack_type' if dataset=='edgeiiot' else 'category' if dataset=='botiot_alternative' else 'attack_cat'
    numeric=[c for c in raw.select_dtypes(include=np.number).columns if c not in ['id','label','Attack_label']]
    if dataset=='botiot_alternative':
        from botiot_data import FEATURES
        numeric=FEATURES
    if dataset=='unsw':assert len(raw)==82332 and len(numeric)==39,(len(raw),len(numeric))
    dedup=raw[numeric+[target]].drop_duplicates()
    if dataset=='unsw':assert dedup[target].value_counts().to_dict()==EXPECTED,'Dataset does not match paper Table 4'
    if dataset=='edgeiiot':assert len(raw)==157800 and len(dedup)==152245,'Edge-IIoTset count mismatch'
    source_rows=dedup.index.to_numpy();data=dedup.reset_index(drop=True)
    encoder=LabelEncoder().fit(data[target])
    y=encoder.transform(data[target])
    train,test=train_test_split(np.arange(len(data)),test_size=.3,stratify=y,random_state=SEED)
    assert not np.intersect1d(train,test).size
    np.savez(out/'split.npz',train=train,test=test,original_csv_rows=source_rows)
    X=data[numeric].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
    keep=np.nanstd(X[train],axis=0)>0
    numeric=np.array(numeric)[keep];X=X[:,keep]
    imputer=SimpleImputer(strategy='median').fit(X[train])
    scaler=StandardScaler().fit(imputer.transform(X[train]))
    Xtr=scaler.transform(imputer.transform(X[train]));Xte=scaler.transform(imputer.transform(X[test]))
    mi=mutual_info_classif(Xtr,y[train],random_state=SEED)
    top=np.argsort(mi)[-min(30,len(mi)):]
    # Paper specifies a tree estimator but not its type/parameters. This is our explicit choice.
    rfe=RFE(RandomForestClassifier(n_estimators=100,max_depth=18,n_jobs=1,random_state=SEED),
            n_features_to_select=min(20,len(top)),step=2).fit(Xtr[:,top],y[train])
    selected=top[rfe.support_]
    if feature_mode=='paper': selected=np.array([np.where(numeric==n)[0][0] for n in PAPER_FEATURES])
    preprocessing=dict(numeric_columns=numeric.tolist(),imputer=imputer,scaler=scaler,
        selected_indices=selected,encoder=encoder,rfe=rfe)
    joblib.dump(preprocessing,out/'preprocessing.joblib')
    pd.DataFrame({'feature':numeric,'mutual_information':mi,'selected':np.isin(np.arange(len(mi)),selected)}).to_csv(out/'features.csv',index=False)
    metadata=dict(raw_rows=len(raw),deduplicated_rows=len(data),train_rows=len(train),test_rows=len(test),
        counts=data[target].value_counts().to_dict(),seed=SEED,dataset=dataset,deduplication='numeric predictors plus class target, excluding ID, binary labels and categorical predictors',
        raw_numeric_features=len([c for c in raw.select_dtypes(include=np.number).columns if c not in ['id','label','Attack_label']]),nonconstant_numeric_features=len(numeric),feature_mode=feature_mode,
        selected_features=numeric[selected].tolist(),class_names=encoder.classes_.tolist(),
        source_sha256=hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
        rare_class_definition='three rarest classes by training support',
        training_counts={str(k):int(v) for k,v in zip(encoder.classes_,np.bincount(y[train]))})
    write_json(out/'dataset.json',metadata)
    return Xtr[:,selected],Xte[:,selected],y[train],y[test],encoder,preprocessing


def group_split(X,y,normal_id):
    from scipy.sparse.csgraph import connected_components
    groups=np.empty(len(y),dtype=int);offset=0
    for c in np.unique(y):
        ids=np.flatnonzero(y==c)
        graph=radius_neighbors_graph(X[ids],radius=.01,mode='connectivity',include_self=True,n_jobs=1)
        n,local=connected_components(graph,directed=False)
        groups[ids]=local+offset;offset+=n
    candidates=[]
    # 3 of 10 folds approximate 30%; pick by class proportions only, never model scores.
    folds=list(StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=SEED).split(X,y,groups))
    import itertools
    for comb in itertools.combinations(range(10),3):
        test=np.sort(np.concatenate([folds[i][1] for i in comb]))
        fractions=np.bincount(y[test],minlength=len(np.unique(y)))/np.bincount(y)
        candidates.append((np.max(np.abs(fractions-.3)),test))
    error,test=min(candidates,key=lambda p:p[0]);train=np.setdiff1d(np.arange(len(y)),test)
    assert not np.intersect1d(groups[train],groups[test]).size
    return train,test,groups,error


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--features',choices=['hybrid','paper'],default='hybrid')
    parser.add_argument('--dataset',choices=['unsw','edgeiiot','botiot_alternative'],default='unsw')
    parser.add_argument('--models',default='all')
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    for d in ['metrics','per_class','models','predictions','figures']: (out/d).mkdir(exist_ok=True)
    import importlib.metadata as metadata
    write_json(out/'runtime.json',dict(python=platform.python_version(),platform=platform.platform(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),thread_limit=1,
        packages={p:metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','imbalanced-learn','xgboost','lightgbm','shap']}))
    results=[];corruptions=[];binary_results=[]
    with threadpool_limits(limits=1):
        print('Preparing dataset and train-only MI/RFE selection',flush=True)
        Xtr,Xte,ytr,yte,enc,pre=prepare(args.data,out,args.features,args.dataset)
        normal_id=int(enc.transform(['Normal'])[0]);labels=np.arange(len(enc.classes_))
        rare_ids=np.argsort(np.bincount(ytr))[:3]
        low,high=Xtr.min(axis=0),Xtr.max(axis=0)
        Xbal,ybal,wbal=balanced_data(Xtr,ytr,True,True)
        config={'EdgeGuard':EdgeGuard(normal_id),**baseline_models()}
        requested=list(config) if args.models=='all' else args.models.split(',')
        for name in requested:
            print('Training',name,flush=True);model=config[name];t=time.perf_counter()
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                if name=='EdgeGuard': model.fit(Xtr,ytr)
                else: fit_weighted(model,Xbal,ybal,wbal)
            elapsed=time.perf_counter()-t
            record,rows=evaluate(name,model,Xte,yte,enc,rare_ids,out,low,high,robust=name!='kNN')
            record['fit_seconds']=elapsed;record['fit_warnings']=list(dict.fromkeys(str(w.message) for w in ws))
            record['sample_weight_supported']='sample_weight' in inspect.signature(model.fit).parameters or name=='EdgeGuard'
            write_json(out/'metrics'/f'{name}.json',record)
            results.append(record);corruptions.extend(rows)
            pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
            pd.DataFrame(corruptions).to_csv(out/'corruptions.csv',index=False)
            if name!='EdgeGuard':
                binary_model=baseline_models()[name];yb=(ytr!=normal_id).astype(int)
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always')
                    fit_weighted(binary_model,Xtr,yb,compute_sample_weight('balanced',yb))
                bp=binary_model.predict(Xte);ytb=(yte!=normal_id).astype(int)
                score=binary_model.predict_proba(Xte)[:,1] if hasattr(binary_model,'predict_proba') else binary_model.decision_function(Xte)
                binary_results.append(dict(model=name,accuracy=accuracy_score(ytb,bp),
                    weighted_f1=f1_score(ytb,bp,average='weighted'),attack_f1=f1_score(ytb,bp),
                    attack_recall=recall_score(ytb,bp),roc_auc=roc_auc_score(ytb,score),
                    fit_warnings=list(dict.fromkeys(str(w.message) for w in ws))))
                pd.DataFrame(binary_results).to_csv(out/'binary_baselines.csv',index=False)
        if args.extended:
            ablations={
                'Hierarchical_unbalanced_noaug':EdgeGuard(normal_id,False,False,False),
                'Hierarchical_balanced_noaug':EdgeGuard(normal_id,True,True,False),
                'Augmented_unbalanced':EdgeGuard(normal_id,False,False,True),
                'Augmented_weights_only':EdgeGuard(normal_id,False,True,True),
                'Augmented_SMOTE_only':EdgeGuard(normal_id,True,False,True),
                'Flat_unbalanced_noaug':hist(),
            }
            for name,model in ablations.items():
                print('Ablation',name,flush=True);t=time.perf_counter();model.fit(Xtr,ytr)
                fit_seconds=time.perf_counter()-t
                record,rows=evaluate(name,model,Xte,yte,enc,rare_ids,out,low,high)
                record['fit_seconds']=fit_seconds;results.append(record);corruptions.extend(rows)
                pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
                pd.DataFrame(corruptions).to_csv(out/'corruptions.csv',index=False)
            # Stricter group split: fit imputation/scaling on its own training rows afterwards.
            # Group definition uses original training scaler, as in the paper's sensitivity analysis.
            X=np.concatenate([Xtr,Xte]);y=np.concatenate([ytr,yte])
            tr,te,g,error=group_split(X,y,normal_id)
            split=np.load(out/'split.npz');original_ids=np.concatenate([split['train'],split['test']])
            np.savez(out/'group_split.npz',train_dedup_indices=original_ids[tr],test_dedup_indices=original_ids[te],groups=g)
            # Refit scaler using only new train rows. Undo old selected scaling first.
            selected=pre['selected_indices'];orig=X*pre['scaler'].scale_[selected]+pre['scaler'].mean_[selected]
            group_scaler=StandardScaler().fit(orig[tr]);gtr=group_scaler.transform(orig[tr]);gte=group_scaler.transform(orig[te])
            print('Group-aware split',len(tr),len(te),'max deviation',error,flush=True)
            model=EdgeGuard(normal_id).fit(gtr,y[tr])
            record,rows=evaluate('GroupAware_EdgeGuard',model,gte,y[te],enc,rare_ids,out,gtr.min(axis=0),gtr.max(axis=0))
            record.update(train_rows=len(tr),test_rows=len(te),max_class_test_fraction_deviation=error)
            write_json(out/'metrics'/'GroupAware_EdgeGuard.json',record)
            results.append(record);corruptions.extend(rows)
        pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
        pd.DataFrame(corruptions).to_csv(out/'corruptions.csv',index=False)
        if len(requested)>=3:
            matrix=np.array([[json.loads((out/'per_class'/f'{name}.json').read_text())[c]['f1-score'] for name in requested] for c in enc.classes_])
            stat,p=friedmanchisquare(*matrix.T);ranks=rankdata(-matrix,axis=1).mean(axis=0)
            k=len(requested);blocks=len(enc.classes_)
            cd=float(studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*blocks)))
            write_json(out/'statistics.json',dict(friedman_statistic=stat,p_value=p,nemenyi_cd_alpha_05=cd,
                blocks='ten per-class F1 scores from one held-out split; not independent repeated trials',
                average_ranks=dict(zip(requested,ranks)),
                interpretation='Failure to reject a pairwise difference does not establish equivalence or non-inferiority.'))
        # Explain each hierarchy decision separately. Stage 1 is binary; stage 2 is nine-way.
        if 'EdgeGuard' in requested:
            import shap
            model=joblib.load(out/'models'/'EdgeGuard.joblib')
            rng=np.random.default_rng(SEED)
            ids=np.sort(rng.choice(len(Xte),min(256,len(Xte)),replace=False))
            for stage,est in [('stage1',model.stage1_),('stage2',model.stage2_)]:
                explanation=shap.TreeExplainer(est)(Xte[ids])
                values=explanation.values;np.savez_compressed(out/f'shap_{stage}.npz',values=values,test_offsets=ids,base_values=explanation.base_values)
                axes=(0,) if values.ndim==2 else (0,2)
                importance=np.abs(values).mean(axis=axes)
                pd.DataFrame({'feature':np.array(pre['numeric_columns'])[pre['selected_indices']],
                    'mean_absolute_shap':importance}).sort_values('mean_absolute_shap',ascending=False).to_csv(out/f'shap_{stage}.csv',index=False)
            print('SHAP: both stages explained on 256 held-out flows',flush=True)
        print('EXPERIMENT COMPLETE',out,flush=True)


if __name__=='__main__': main()
