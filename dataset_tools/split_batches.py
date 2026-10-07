"""Summarize complete, mixed and partial batches using a verified manifest."""
import argparse
from pathlib import Path

import pandas as pd
from dataset_manifest import read_manifest
from kswin_compat import positive_integer

BASE = Path(__file__).resolve().parent


def summarize_batches(frame, metadata, batch_size=2000):
    positive_integer(batch_size,'batch_size')
    rows=[]
    for batch_id,start in enumerate(range(0,len(frame),batch_size)):
        end=min(start+batch_size,len(frame))
        batch=frame.iloc[start:end]
        names=[s['name'] for s in metadata['sources'] if s['start_row']<end and s['end_row_exclusive']>start]
        attack=int((batch.attack_flag==1).sum()) if 'attack_flag' in batch else None
        rows.append(dict(batch_id=batch_id,batch_number1=batch_id+1,start_row=start,
                         end_row_exclusive=end,source=names[0] if len(names)==1 else 'Mixed',
                         rows=len(batch),partial=len(batch)<batch_size,attack=attack,
                         attack_ratio=attack/len(batch) if attack is not None else None))
    return pd.DataFrame(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=BASE/'ImprTotal.csv')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--config',type=Path,help='optional runner config containing batch_size')
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--output',type=Path,default=BASE/'generated'/'batch_summary.csv')
    args=parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError(args.output)
    if args.config:
        from mpcawin_run import load_configs
        configs=load_configs(args.config)
        if len(configs)!=1: parser.error('use a single configuration')
        size=configs[0]['batch_size']
        if args.batch_size is not None and args.batch_size!=size:
            parser.error('batch-size override contradicts config')
    else: size=args.batch_size or 2000
    metadata=read_manifest(args.data,args.manifest)
    frame=pd.read_csv(args.data)
    if len(frame)!=metadata['data']['rows']: raise ValueError('manifest row count mismatch')
    table=summarize_batches(frame,metadata,size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output,index=False)
    print(table.to_string(index=False))


if __name__=='__main__': main()
