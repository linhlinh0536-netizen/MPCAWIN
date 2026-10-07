"""Full-stream coefficient/seed sensitivity using the same pipeline as the runner."""
import argparse
import json
from pathlib import Path

import pandas as pd
from dataset_manifest import read_manifest, select_features
from mpcawin_run import load_configs, run_stream, runtime_provenance


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20])
    parser.add_argument('--coefficients',type=float,nargs='+',default=[2,3,4,5,6])
    parser.add_argument('--output-dir',type=Path,default=Path('results/sensitivity'))
    args=parser.parse_args(argv)
    configs=load_configs(args.config)
    if len(configs)!=1: parser.error('choose one configuration; compare backends in separate sweeps')
    metadata=read_manifest(args.data,args.manifest)
    frame=pd.read_csv(args.data)
    if len(frame)!=metadata['data']['rows'] or frame.columns.tolist()!=metadata['data']['columns']:
        raise ValueError('dataset shape/schema mismatch')
    features=select_features(frame,metadata['feature_columns'])
    boundaries=[b['row'] for b in metadata['boundaries']]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    for coefficient in args.coefficients:
        for seed in args.seeds:
            config=dict(configs[0],coef_std=coefficient,seed=seed)
            table,summary=run_stream(features,config,boundaries,metadata['sources'])
            name=f"{config['name']}_c{coefficient:g}_seed{seed}"
            table.to_csv(args.output_dir/f'{name}.csv',index=False)
            evaluation=summary['evaluation']
            boundary_batch=boundaries[0]//config['batch_size'] if boundaries else len(table)
            before=table.loc[table.batch_id<boundary_batch,'alert'].dropna()
            after=table.loc[table.batch_id>=boundary_batch,'alert'].dropna()
            record=dict(configuration=config['name'],c=coefficient,seed=seed,
                        pre_alerts=int(before.sum()),pre_evaluated=len(before),
                        post_alerts=int(after.sum()),post_evaluated=len(after),
                        warmup_batches=summary['warmup_batches'],
                        concept_reset_count=summary['concept_reset_count'],
                        misses=sum(match['missed'] for match in evaluation['matches']),
                        detections=json.dumps(summary['detected_batches']),
                        pre_boundary_events=len(evaluation['detections_before_first_boundary']),
                        unmatched_events=json.dumps(evaluation['unmatched_detection_batches']),
                        index_delays=json.dumps([m['index_delay_batches'] for m in evaluation['matches']]))
            records.append(record)
            print(json.dumps(record),flush=True)
    pd.DataFrame(records).to_csv(args.output_dir/'summary.csv',index=False)
    (args.output_dir/'metadata.json').write_text(json.dumps(dict(
        provenance=runtime_provenance(args.data,args.manifest or args.data.with_suffix('.manifest.json'),features),
        base_config=configs[0],
        seeds=args.seeds,coefficients=args.coefficients),indent=2)+'\n')


if __name__=='__main__': main()
