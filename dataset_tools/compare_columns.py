"""Export ordered feature/schema intersections without changing datasets."""
import argparse
import json
from pathlib import Path

import pandas as pd
from dataset_manifest import EXCLUDED_COLUMNS

BASE = Path(__file__).resolve().parent


def compare(first,second):
    a=pd.read_csv(first,nrows=1000)
    b=pd.read_csv(second,nrows=1000)
    common=[c for c in a if c in b]
    return dict(columns_equal=a.columns.tolist()==b.columns.tolist(),ordered_intersection=common,
                numerical_feature_candidates=[c for c in common if c not in EXCLUDED_COLUMNS
                    and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])],
                excluded=[c for c in common if c in EXCLUDED_COLUMNS],
                dtype_sample_rows=1000,dtypes_first=a.dtypes.astype(str).to_dict(),
                dtypes_second=b.dtypes.astype(str).to_dict(),
                only_first=[c for c in a if c not in b],only_second=[c for c in b if c not in a])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,default=BASE/'cicids2017-imp'/'Reformatted_CIC-IDS2017-Imp.csv')
    parser.add_argument('--second',type=Path,default=BASE/'cicids2018-imp'/'Reformatted_CSE-CIC-IDS2018-Imp.csv')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    result=compare(args.first,args.second)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        if args.output.exists(): raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text)
    print(text)


if __name__=='__main__': main()
