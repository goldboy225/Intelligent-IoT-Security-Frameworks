"""Record real access outcomes and source metadata; do not fabricate missing data."""
import hashlib
import json
from pathlib import Path
import requests
from fetch_data import COMMIT,SHA,ROOT


def main():
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=True)
    urls=[
        'https://zenodo.org/api/records/15168928',
        'https://zenodo.org/api/records/18467070',
        'https://sherlock.wattson.it/download/',
        f'https://media.githubusercontent.com/media/ipal-ids/ipal_datasets/{COMMIT}/Sherlock/ipal/01-Basic/test.n302.state.gz',
        'https://huggingface.co/distilbert/distilbert-base-uncased/resolve/main/config.json']
    status=[]
    for url in urls:
        try:
            response=requests.get(url,timeout=12,stream=True)
            status.append(dict(url=url,http_status=response.status_code));response.close()
        except requests.RequestException as e:status.append(dict(url=url,error_type=type(e).__name__))
    catalog_url=f'https://raw.githubusercontent.com/ipal-ids/ipal_datasets/{COMMIT}/Sherlock/ipal/01-Basic/test.events.json'
    r=requests.get(catalog_url,timeout=20);r.raise_for_status();events=r.json()
    evidence=dict(source_url=catalog_url,source_commit=COMMIT,sha256=hashlib.sha256(r.content).hexdigest(),
        event_count=len(events),families=sorted(set(e['description'].split(':')[0] for e in events)),
        generator_control_events=[{k:e[k] for k in ['id','description','start','end']} for e in events if 'Generator Control' in e['description']],
        benign_catalog_events=sum('(benign event)' in str(e['id']) for e in events),
        original_paper_source='https://github.com/ipal-ids/ipal_datasets/blob/'+COMMIT+'/Sherlock/documentation/paper.pdf',
        original_paper_evidence='Sections 3.4 and 3.5 describe four attack types and require maintenance events to be counted as normal; Section 3.6 logs network-observed state every second, distinct from 10-second measurement transmission.',
        interpretation='A six-class task adding GeneratorControl cannot be reconstructed from this catalog without an additional author-supplied label mapping or custom data.')
    (out/'dataset_source_audit.json').write_text(json.dumps(evidence,indent=2))
    completed_runs=list(Path('/workspace/sherlock-review/runs').glob('*/results.json'))
    result=dict(status='blocked_real_dataset_training' if not (ROOT/'test.n302.state.gz').is_file() else 'data_present_label_provenance_still_requires_checking',checks=status,
        real_state_file_present=(ROOT/'test.n302.state.gz').is_file(),
        required_state_sha256=SHA,
        historical_native_git_read_in_this_review='public IPAL source checkout succeeded at pinned commit',
        historical_native_git_lfs_result_in_this_review='Resource not accessible by integration; Git LFS pull exit 2',
        public_author_code_search='No matching repository found by title/architecture queries; absence from search is not proof that no code exists.',
        dataset_version='Manuscript unspecified. Public IPAL mirror pinned by commit/hash; Zenodo v1 and v3 identified, not silently treated as identical.',
        completed_training_result_files_in_standard_run_directory=len(completed_runs),
        next_actions=['Apply saved network domain additions in environment settings and retry data/model access.',
            'Obtain or clarify six-class labels, exact feature table/splits, checkpoint and architecture choices from the authors.'],
        already_completed=['confusion matrix arithmetic','public event-label audit','architecture structural checks','full-width DistilBERT code tests'])
    (out/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
