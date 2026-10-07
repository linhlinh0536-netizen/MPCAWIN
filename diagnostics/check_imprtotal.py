"""Run the labeled experimental configurations using the installed repaired code."""
import argparse
from pathlib import Path

from mpcawin_run import main as run_main

BASE=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--out-dir',type=Path,default=BASE/'diagnostics'/'repaired_results')
    args=parser.parse_args()
    command=['--data',str(args.data),'--config',str(BASE/'configs'/'experiments.json'),
             '--output-dir',str(args.out_dir)]
    if args.manifest: command+=['--manifest',str(args.manifest)]
    return run_main(command)


if __name__=='__main__': raise SystemExit(main())
