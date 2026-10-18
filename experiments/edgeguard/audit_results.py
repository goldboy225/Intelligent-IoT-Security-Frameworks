"""Cross-check persisted predictions, metrics, split separation, and model loading."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
# Also supplies the legacy __main__.EdgeGuard name for the initial run's pickle.
from run_experiment import EdgeGuard,write_json


def main():
    root=Path('/workspace/edgeguard');checks=[]
    for folder,csv,target in [
        ('hybrid','UNSW_NB15_testing-set.csv','attack_cat'),
        ('paper_features','UNSW_NB15_testing-set.csv','attack_cat'),
        ('edgeiiot','ML-EdgeIIoT-dataset.csv','Attack_type'),
        ('botiot_alternative','Bot-IoT-5percent-independent-sample.csv','category')]:
        result=root/'results'/folder
        assert 'EXPERIMENT COMPLETE' in (result/'run.log').read_text(),folder
        pre=joblib.load(result/'preprocessing.joblib');split=np.load(result/'split.npz')
        raw=pd.read_csv(root/'data'/csv,low_memory=False)
        if folder=='edgeiiot':
            columns=[c for c in raw.select_dtypes(include=np.number).columns if c!='Attack_label']
        else:columns=pre['numeric_columns']
        data=raw[columns+[target]].drop_duplicates().reset_index(drop=True)
        y=pre['encoder'].transform(data[target]);tr,te=split['train'],split['test']
        assert not np.intersect1d(tr,te).size
        assert len(tr)+len(te)==len(data)
        assert len(np.unique(np.r_[tr,te]))==len(data)
        X=pre['imputer'].transform(data[pre['numeric_columns']].to_numpy())
        np.testing.assert_allclose(X[tr].mean(axis=0),pre['scaler'].mean_,rtol=1e-12,atol=1e-10)
        assert (result/'features.csv').exists()
        columns=pre['numeric_columns'];hashes=pd.util.hash_pandas_object(data[columns],index=False).to_numpy()
        overlap=len(np.intersect1d(hashes[tr],hashes[te]))
        summary=pd.read_csv(result/'summary.csv')
        for row in summary.itertuples():
            test=te
            if row.model=='GroupAware_EdgeGuard':test=np.load(result/'group_split.npz')['test_dedup_indices']
            pred=np.load(result/'predictions'/f'{row.model}.npy')
            assert len(pred)==len(test),(folder,row.model)
            np.testing.assert_allclose(accuracy_score(y[test],pred),row.accuracy,atol=1e-14)
            np.testing.assert_allclose(f1_score(y[test],pred,average='macro',zero_division=0),row.macro_f1,atol=1e-14)
            modelpath=result/'models'/f'{row.model}.joblib'
            model=joblib.load(modelpath)
            # Store hierarchy class under its importable module for future independent reloads.
            if isinstance(model,EdgeGuard):
                joblib.dump(model,modelpath,compress=0)
                metric=result/'metrics'/f'{row.model}.json'
                info=json.loads(metric.read_text());info['serialized_bytes']=modelpath.stat().st_size
                write_json(metric,info)
                summary.loc[summary.model==row.model,'serialized_bytes']=modelpath.stat().st_size
            joblib.load(modelpath)
        summary.to_csv(result/'summary.csv',index=False)
        corruption=pd.read_csv(result/'corruptions.csv')
        for name,block in corruption.groupby('model'):
            assert len(block)==12 and not block[['family','level']].duplicated().any()
            clean=float(summary.set_index('model').loc[name,'macro_f1'])
            np.testing.assert_allclose(block.retention_ratio,block.macro_f1/clean,atol=1e-14)
        checks.append(dict(dataset=folder,models_checked=len(summary),heldout_predictions_match_metrics=True,
            train_test_indices_disjoint=True,training_scaler_verified=True,
            exact_numeric_predictor_hash_overlap=overlap,corruption_conditions_checked=len(corruption),
            model_reload_passed=True))
    write_json(root/'results/audit.json',dict(status='passed',checks=checks,
        note='Predictor overlap counts exclude target; numeric+target deduplication can retain conflicting-label predictors. A zero overlap does not rule out near duplicates.'))
    print(json.dumps(checks,indent=2));print('AUDIT PASSED')


if __name__=='__main__':main()
