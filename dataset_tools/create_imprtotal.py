"""Concatenate source subsets in order, or describe an existing ImprTotal CSV."""
import argparse
from pathlib import Path

import pandas as pd
from dataset_manifest import describe_concatenation, write_manifest

BASE = Path(__file__).resolve().parent


def create_total(file_2017, file_2018, output, describe_existing=False):
    paths = [Path(file_2017), Path(file_2018)]
    output = Path(output)
    if describe_existing:
        if not output.exists():
            raise FileNotFoundError(output)
    else:
        if output.exists() or output.with_suffix('.manifest.json').exists():
            raise FileExistsError(f'refusing to overwrite {output}')
        first, second = [pd.read_csv(path) for path in paths]
        if first.columns.tolist() != second.columns.tolist():
            raise ValueError('source schemas and column order must match')
        if not len(first) or not len(second):
            raise ValueError('source subsets must be nonempty')
        total = pd.concat([first, second], ignore_index=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        total.to_csv(output, index=False)
        del first, second, total
    metadata = describe_concatenation(output, paths, ['Impr2017','Impr2018'])
    path = write_manifest(output, metadata)
    print(f"Rows: {metadata['data']['rows']}; boundary: {metadata['boundaries'][0]['row']}; manifest: {path}")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file-2017', type=Path, default=BASE/'Impr2017_stratified.csv')
    parser.add_argument('--file-2018', type=Path, default=BASE/'Impr2018_stratified.csv')
    parser.add_argument('--output', type=Path, default=BASE/'generated'/'ImprTotal.csv')
    parser.add_argument('--describe-existing', action='store_true')
    args = parser.parse_args(argv)
    create_total(args.file_2017,args.file_2018,args.output,args.describe_existing)


if __name__ == '__main__':
    main()
