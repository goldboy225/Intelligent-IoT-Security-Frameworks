"""Recompute reported numbers, not a trained-model reproduction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def main():
    root=Path(__file__).parent;out=root/'results';out.mkdir(exist_ok=True)
    facts=json.loads((root/'paper_facts.json').read_text());matrix=np.array(facts['figure5_confusion_matrix'])
    true=[];pred=[]
    for i,j in np.ndindex(matrix.shape):
        true.extend([i]*int(matrix[i,j]));pred.extend([j]*int(matrix[i,j]))
    true=np.array(true);pred=np.array(pred)
    report=classification_report(true,pred,target_names=facts['class_order'],output_dict=True,zero_division=0)
    pd.DataFrame(matrix,index=facts['class_order'],columns=facts['class_order']).to_csv(out/'transcribed_confusion_matrix.csv')
    pd.DataFrame(report).T.to_csv(out/'recomputed_classification_report.csv')
    benign=facts['class_order'].index('Benign')
    binary=confusion_matrix(true!=benign,pred!=benign)
    result={
        'status':'arithmetic_audit_completed_not_training_validation',
        'evaluation_samples':int(matrix.sum()),'correct_predictions':int(np.trace(matrix)),
        'accuracy_percent':float(np.trace(matrix)/matrix.sum()*100),
        'macro_f1':report['macro avg']['f1-score'],'weighted_f1':report['weighted avg']['f1-score'],
        'accuracy_rounds_to_reported_two_decimal_percent':round(report['accuracy']*100,2)==facts['reported_accuracy_percent'],
        'macro_f1_rounds_to_reported_one_decimal_percent':round(report['macro avg']['f1-score']*100,1)==facts['reported_macro_f1_percent'],
        'macro_f1_rounds_to_table_two_decimal_fraction':round(report['macro avg']['f1-score'],2)==.96,
        'rounding_interpretation':'Macro F1 is 95.7623%, or 95.8% to one decimal. It agrees with Table V F1=0.96 rounded as a fraction. This precision difference alone is not a substantive numerical contradiction.',
        'figure4_equals_figure5_row_totals':np.array_equal(matrix.sum(axis=1),facts['figure4_counts']),
        'total_required_if_matrix_is_20_percent_test_split':int(matrix.sum()*5),
        'split_count_caveat':'8641 test rows are also compatible with ceil(0.2*43204). Equality with Figure 4 is not proof of train/test leakage; Figure 4 may show test-only counts. Clarification required.',
        'binary_metrics_from_collapsing_multiclass_matrix_not_separate_binary_head':{
            'confusion_matrix':binary.tolist(),'attack_precision':float(binary[1,1]/binary[:,1].sum()),
            'attack_recall':float(binary[1,1]/binary[1].sum()),'attack_f1':f1_score(true!=benign,pred!=benign)},
        'attack_family_confusions_excluding_benign':int(matrix[np.ix_(np.arange(6)!=benign,np.arange(6)!=benign)].sum()-np.trace(matrix[np.ix_(np.arange(6)!=benign,np.arange(6)!=benign)])),
        'lowest_class_f1':sorted([(k,v['f1-score']) for k,v in report.items() if k in facts['class_order']],key=lambda x:x[1])[:2],
        'textual_check':'Text says lowest F1s are Industroyer and Control&Freeze; matrix instead gives Industroyer and GeneratorControl.',
        'f1_gain_over_without_attention_percentage_points':100*(.96-.93),
        'attention_gain_text_caveat':'Text claims 2.0 F1 percentage points; Table V implies 3.0.',
        'roc_auc_status':'not independently recoverable from hard confusion counts; requires held-out score vectors',
        'training_status':'not_run_until_real_data_and_architecture_choices_are_available'}
    (out/'paper_arithmetic_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,7));normalized=matrix/matrix.sum(axis=1,keepdims=True)
    ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
    for i,j in np.ndindex(matrix.shape):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if normalized[i,j]>.5 else 'black')
    ax.set_xticks(np.arange(6),facts['class_order'],rotation=40,ha='right');ax.set_yticks(np.arange(6),facts['class_order'])
    ax.set(title='Manuscript confusion matrix: independently recomputed\nTranscribed published counts, not a model rerun',xlabel='Predicted',ylabel='Actual')
    fig.tight_layout();(root/'figures').mkdir(exist_ok=True);fig.savefig(root/'figures/paper_matrix_audit.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
