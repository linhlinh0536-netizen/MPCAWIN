"""Compare repaired paper baseline, scaling, solver, and first-event seed replay."""
import argparse
import json
from pathlib import Path

import pandas as pd
from dataset_manifest import read_manifest,select_features
from kswin_compat import KSWIN
from mpcawin_run import PAPER,run_stream

BASE=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--out-dir',type=Path,default=BASE/'diagnostics'/'repaired_paper_results')
    args=parser.parse_args(argv)
    metadata=read_manifest(args.data,args.manifest)
    features=select_features(pd.read_csv(args.data),metadata['feature_columns'])
    boundaries=[b['row'] for b in metadata['boundaries']]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    summaries=[]
    original=None
    for name,overrides in [('paper',{}),('batch_scaler',dict(scaling='batch')),('full_pca',dict(pca_solver='full'))]:
        config=dict(PAPER,name=name,**overrides)
        table,summary=run_stream(features,config,boundaries,metadata['sources'])
        table.to_csv(args.out_dir/f'{name}.csv',index=False)
        summaries.append(summary)
        if name=='paper': original=table
        print(f"{name}: {summary['detected_batches']}",flush=True)
    events=[]
    alerts=original.loc[original.alert.notna(),['batch_id','alert']]
    for seed in range(50):
        detector=KSWIN(alpha=.009,window_size=20,stat_size=5,random_state=seed)
        first=None
        for row in alerts.itertuples():
            detector.add_element(row.alert)
            if detector.detected_change(): first=int(row.batch_id); break
        events.append(dict(seed=seed,first_detection=first))
    (args.out_dir/'summary.json').write_text(json.dumps(dict(
        full_stream=summaries,first_event_replay=events,
        replay_scope='confirmation only, valid until the first reset'),indent=2)+'\n')


if __name__=='__main__': main()
