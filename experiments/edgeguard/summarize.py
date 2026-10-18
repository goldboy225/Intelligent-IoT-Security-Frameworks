"""Produce a portable report from completed experiment outputs, not invented scores."""
import argparse
import base64
import html
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/workspace/edgeguard'))
    args=parser.parse_args();root=args.root;out=root/'results/hybrid'
    assert 'EXPERIMENT COMPLETE' in (out/'run.log').read_text(),'Primary experiment has not completed'
    share=root/'share/EdgeGuard-IoT-experiment';share.mkdir(parents=True,exist_ok=True)
    tables=share/'results';tables.mkdir(exist_ok=True)
    figures=share/'figures';figures.mkdir(exist_ok=True)
    summary=pd.read_csv(out/'summary.csv');corruptions=pd.read_csv(out/'corruptions.csv')
    dataset=json.loads((out/'dataset.json').read_text())
    names=['EdgeGuard','HistGBM','XGBoost','LightGBM','RandomForest','ExtraTrees','DecisionTree',
           'MLP','kNN','AdaBoost','LogReg','LinearSVC','GaussianNB']
    assert set(names).issubset(summary.model)
    edge=summary.set_index('model').loc['EdgeGuard']
    compare=pd.DataFrame({
        'metric':['Accuracy','Weighted F1','Macro F1','MCC','Mean corrupted macro F1'],
        'paper':[.861,.870,.560,.767,.338],
        'this_run':[edge.accuracy,edge.weighted_f1,edge.macro_f1,edge.mcc,edge.mean_retained_macro_f1]})
    compare['difference']=compare.this_run-compare.paper
    compare.to_csv(tables/'paper_comparison.csv',index=False)
    for filename in ['summary.csv','corruptions.csv','binary_baselines.csv','dataset.json','runtime.json',
                     'statistics.json','features.csv','shap_stage1.csv','shap_stage2.csv','run.log']:
        shutil.copy2(out/filename,tables/filename)
    for dirname in ['per_class','metrics']:
        shutil.copytree(out/dirname,tables/dirname,dirs_exist_ok=True)
    if (root/'data/availability.json').exists():shutil.copy2(root/'data/availability.json',tables/'data_availability.json')
    paper=root/'results/paper_features'
    if (paper/'summary.csv').exists():
        shutil.copy2(paper/'summary.csv',tables/'paper_features_sensitivity.csv')
    diagnostic=root/'results/histgb_diagnostic'
    if diagnostic.exists():shutil.copytree(diagnostic,tables/'histgb_diagnostic',dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('*.joblib','*.npy'))
    additional=[]
    for folder,condition,reference in [('edgeiiot','Edge-IIoTset retraining',.930),
            ('botiot_alternative','Bot-IoT alternative 5% sample (not paper sample)',np.nan)]:
        result_dir=root/'results'/folder
        assert 'EXPERIMENT COMPLETE' in (result_dir/'run.log').read_text(),folder+' not completed'
        row=pd.read_csv(result_dir/'summary.csv').iloc[0]
        additional.append(dict(condition=condition,accuracy=row.accuracy,macro_f1=row.macro_f1,
            paper_macro_f1=reference,mean_retained_macro_f1=row.mean_retained_macro_f1))
        target=tables/folder;target.mkdir(exist_ok=True)
        for name in ['summary.csv','dataset.json','features.csv','runtime.json','corruptions.csv','shap_stage1.csv','shap_stage2.csv','run.log']:
            shutil.copy2(result_dir/name,target/name)
        shutil.copytree(result_dir/'per_class',target/'per_class',dirs_exist_ok=True)
    additional=pd.DataFrame(additional);additional.to_csv(tables/'additional_datasets.csv',index=False)
    transfer=json.loads((root/'results/botiot_transfer/results.json').read_text())
    audit=json.loads((root/'results/audit.json').read_text())
    shutil.copy2(root/'results/audit.json',tables/'audit.json')
    shutil.copy2(root/'results/botiot_transfer/results.json',tables/'botiot_transfer.json')
    shutil.copy2(root/'data/botiot_sample_manifest.json',tables/'botiot_sample_manifest.json')
    code=Path(__file__).parent
    for path in code.iterdir():
        if path.is_file() and path.suffix in ['.py','.sh','.md','.txt']:shutil.copy2(path,share/path.name)

    baseline=summary[summary.model.isin(names)].sort_values('macro_f1')
    fig,ax=plt.subplots(figsize=(10,6));ax.barh(baseline.model,baseline.macro_f1,
        color=['#ce3b34' if x=='EdgeGuard' else '#3a708e' for x in baseline.model])
    ax.set(xlabel='Macro F1 on held-out real UNSW-NB15 flows',xlim=(0,1),title='Clean classification: independent implementation')
    fig.tight_layout();fig.savefig(figures/'clean_comparison.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4))
    highlighted=['EdgeGuard','RandomForest','XGBoost','LightGBM','HistGBM','MLP','LogReg']
    for ax,family in zip(axes,['noise','dropout','quantization']):
        for name in highlighted:
            d=corruptions[(corruptions.model==name)&(corruptions.family==family)].sort_values('level',ascending=family!='quantization')
            ax.plot(np.arange(len(d)),d.macro_f1,marker='o',label=name,linewidth=2.5 if name=='EdgeGuard' else 1)
            ax.set_xticks(np.arange(len(d)),d.level.map(lambda x:f'{x:g}'))
        ax.set(title=family,xlabel='Corruption intensity (quantization: levels)',ylabel='Absolute macro F1',ylim=(0,.65));ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'robustness.png',dpi=160);plt.close(fig)

    report=json.loads((out/'per_class/EdgeGuard.json').read_text())
    labels=dataset['class_names'];split=np.load(out/'split.npz')
    # Original row indices and targets reconstruct the exact held-out ground truth.
    raw=pd.read_csv(root/'data/UNSW_NB15_testing-set.csv')
    numeric=[c for c in raw.select_dtypes(include=np.number).columns if c not in ['id','label']]
    dedup=raw[numeric+['attack_cat']].drop_duplicates().reset_index(drop=True)
    y=pd.Categorical(dedup.attack_cat,categories=labels).codes[split['test']]
    prediction=np.load(out/'predictions/EdgeGuard.npy')
    cm=confusion_matrix(y,prediction,labels=np.arange(10),normalize='true')
    fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
    ax.set_xticks(np.arange(10),labels,rotation=60,ha='right');ax.set_yticks(np.arange(10),labels)
    for i in range(10):
        for j in range(10):ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if cm[i,j]>.5 else 'black')
    ax.set(xlabel='Predicted class',ylabel='True class',title='EdgeGuard held-out recall matrix')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figures/'confusion_matrix.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,stage in zip(axes,['stage1','stage2']):
        d=pd.read_csv(out/f'shap_{stage}.csv').head(10).iloc[::-1]
        ax.barh(d.feature,d.mean_absolute_shap,color='#3a708e');ax.set(title=f'SHAP: {stage}',xlabel='Mean absolute attribution (raw margin)')
    fig.tight_layout();fig.savefig(figures/'shap.png',dpi=160);plt.close(fig)

    ablation=summary[~summary.model.isin(names)].copy()
    content=f'''# EdgeGuard-IoT experiment results

Independent implementation of Jeribi, DOI 10.1016/j.comnet.2026.112813.

The primary real-data experiment completed on {dataset['deduplicated_rows']:,} deduplicated
UNSW-NB15 flows ({dataset['train_rows']:,} train / {dataset['test_rows']:,} held out).
The raw partition and every deduplicated class count match the paper. The author's
code and frozen split were unavailable, so this is an approximate reproduction.

## Proposed detector

{compare.to_markdown(index=False,floatfmt='.4f')}

The detector routed {edge.escalation_rate:.1%} of held-out flows to stage 2.
Its prediction throughput was {edge.throughput_flows_s:,.0f} flows/s on one cloud CPU
core; uncompressed serialized model size was {edge.serialized_bytes/1e6:.2f} MB.
Cloud timings do not establish physical gateway performance or a 4-GB memory budget.

## Baselines

{baseline[['model','accuracy','macro_f1','weighted_f1','mean_retained_macro_f1']].sort_values('macro_f1',ascending=False).to_markdown(index=False,floatfmt='.4f')}

Robustness is the mean absolute macro F1 over 12 seeded corruption conditions.
kNN corruption tests were excluded, following the paper; NaN is unrun, not zero.
The fixed 200-iteration, L2=0 flat HistGBM has an anomalous weak result. A separate
diagnostic records its held-out score trajectory and raw-margin range, and evaluates
L2=1 without replacing the prespecified baseline. No test-score-based iteration
selection is used for the reported baseline. Convergence warnings are saved verbatim
in the corresponding result JSON, rather than treated as convergence.

## Ablations and split sensitivity

{ablation[['model','accuracy','macro_f1','rare_class_recall','mean_retained_macro_f1']].to_markdown(index=False,floatfmt='.4f')}

Rare recall here averages Worms, Backdoor and Shellcode. The paper does not fully
specify this aggregate, so direct comparison of that column is uncertain.
The group-aware experiment rebuilds its split, so it is reported separately from
same-test-set baseline comparisons.

The weighting-only augmented ablation scores 0.5800 macro F1 and 0.3514 retained
macro F1, slightly above the full detector (0.5702 / 0.3444) in this run, while
the full detector gives higher mean recall on the three rarest classes. Thus the
paper's combined design is not uniformly best on every metric here.

## Audit and data-quality findings

All persisted classification metrics were independently recomputed from saved
held-out predictions. Split indices are disjoint, scalers match their training
rows, corruption conditions are complete, and saved models reload successfully.

However, numeric-predictor duplication remains: {audit['checks'][0]['exact_numeric_predictor_hash_overlap']}
unique predictor hashes occur on both sides of the primary UNSW split after
numeric-plus-target deduplication. Because identical predictors with different
labels remain separate rows, matching the paper's row count does **not** establish
predictor-level separation. Removing the three categorical fields can also make
different original flows indistinguishable numerically. Edge-IIoTset has
{audit['checks'][2]['exact_numeric_predictor_hash_overlap']} shared predictor hashes
on the retained numeric schema. The paper's within-class near-duplicate grouping
does not guarantee separation of identical predictors carrying different labels.
These are explicit limitations of the reproduced protocol; the reported metrics
are not a leakage-free deployment guarantee. A stricter study should group
identical predictors across all labels before splitting and reselect features
within each resulting training split.

## What this does and does not reproduce

Completed: train-only MI/RFE, hierarchy and flat baselines, clean/binary metrics,
corruption sweep, imbalance and component ablations, group-aware split sensitivity,
single-core batch timing, model size, per-stage SHAP, and per-class Friedman/Nemenyi.
Exact hyperparameter/default and augmentation choices are documented in README.md.
Neither similar scores nor a nonsignificant test proves equivalence to the author's
implementation. There is one seeded split, not repeated independent experiments.

## Additional public-data experiments

{additional.to_markdown(index=False,floatfmt='.4f')}

Edge-IIoTset matches the paper's 157,800 raw and 152,245 deduplicated rows.
There are 34 globally variable numeric predictors; train-only constant detection
retains 33 for this split. Independent MI/RFE retains twenty of them.

The five-feature UNSW-to-Bot-IoT transfer flagged {transfer['stage1_attack_recall']:.2%}
of {transfer['target_flows']:,} genuine DoS flows as attacks (paper: 7.3%). Its own
in-domain Stage-1 F1 was {transfer['in_domain_stage1_f1']:.4f} and attack recall was
{transfer['in_domain_stage1_attack_recall']:.4f}. This is attack recall on an
attack-only dataset, **not multiclass macro F1**. It supports severe domain shift,
without claiming numerical identity to the paper's result.

The original forty-scenario-file Bot-IoT sample was not located in the public
sources checked. Its claimed 136,039-row same-dataset result is **not reproduced**.
The alternative Bot-IoT sample uses the four-file 5% release, all benign/Theft rows,
and up to 3,000 random rows per other category per file. It has 18,556 sampled
rows before deduplication and 16,330 afterwards, five classes, fifteen numeric
features; its smaller benign class makes direct comparison inappropriate. The
full sample manifest is included. No synthetic data stand in for any benchmark.

Public GitHub mirrors enabled these experiments despite proxy errors from the
university, IEEE DataPort, Kaggle and UNSW SharePoint sources. The data mirrors
are pinned to commits and hashes, but no independent original-source checksum
was available. No matching public author implementation was located; the similarly
named `raaddi/edgeguard-iot` repository is an unrelated SmartHome project.

Source code, dependency lock, per-class scores, corruption scores and the detailed
protocol are included. Raw benchmark data and trained model binaries remain in the
cloud workspace and are excluded from this shareable report bundle.
'''
    # to_markdown requires tabulate, installed by setup requirements.
    (share/'REPORT.md').write_text(content)
    blocks=['<h1>EdgeGuard-IoT: real-data experiment</h1>',
        '<p>Independent implementation; approximate reproduction. See REPORT.md and README.md for assumptions.</p>',
        '<h2>Paper comparison</h2>',compare.to_html(index=False,float_format=lambda x:f'{x:.4f}'),
        '<h2>Baselines</h2>',baseline[['model','accuracy','macro_f1','mean_retained_macro_f1']].to_html(index=False,float_format=lambda x:f'{x:.4f}'),
        '<h2>Additional public datasets</h2>',additional.to_html(index=False,float_format=lambda x:f'{x:.4f}')]
    for image in sorted(figures.glob('*.png')):
        encoded=base64.b64encode(image.read_bytes()).decode()
        blocks.append(f'<h2>{html.escape(image.stem)}</h2><img src="data:image/png;base64,{encoded}" style="max-width:100%">')
    blocks+=['<h2>Protocol, limits, and detailed results</h2>','<pre>'+html.escape(content)+'</pre>']
    (share/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>EdgeGuard-IoT experiment</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>'+''.join(blocks))
    # Portable notebook has actual outputs from the completed run, embedded with the report.
    import nbformat
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(content),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = Path.cwd()\ndisplay(pd.read_csv(root / 'results/summary.csv'))\nfor name in ['clean_comparison', 'robustness', 'confusion_matrix', 'shap']:\n    display(Image(filename=str(root / 'figures' / (name + '.png'))))")])
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=120,kernel_name='edgeguard',resources={'metadata':{'path':str(share)}}).execute()
    nbformat.write(nb,share/'EdgeGuard_results.executed.ipynb')
    archive=root/'share/EdgeGuard-IoT-experiment.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in share.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(share.parent))
    print('Report:',share/'REPORT.md');print('Download bundle:',archive)


if __name__=='__main__':main()
