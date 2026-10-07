"""Dataset provenance, ordered feature selection and source-change boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDED_COLUMNS = ('id', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp',
                    'Attempted Category', 'attack_name', 'attack_flag', 'attack_step',
                    '__source_row')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(data):
    return Path(data).with_suffix('.manifest.json')


def select_features(frame, columns=None):
    expected = frame.drop(columns=list(EXCLUDED_COLUMNS), errors='ignore').select_dtypes(include=[np.number])
    if columns is not None:
        if not columns or len(columns) != len(set(columns)):
            raise ValueError('manifest feature columns must be nonempty and unique')
        if any(column in EXCLUDED_COLUMNS for column in columns):
            raise ValueError('labels and metadata must not be detector features')
        if not set(columns).issubset(expected.columns):
            raise ValueError('manifest features must exist and be numeric')
        expected = expected.loc[:, columns]
    if expected.empty or not np.isfinite(expected.to_numpy(dtype=float)).all():
        raise ValueError('features must be nonempty, numeric and finite')
    return expected


def read_manifest(data, path=None):
    path = Path(path) if path else manifest_path(data)
    result = json.loads(path.read_text())
    if result.get('schema_version') != 1:
        raise ValueError('unsupported dataset manifest schema')
    if result['data']['sha256'] != sha256_file(data):
        raise ValueError('dataset SHA256 does not match manifest')
    rows = result['data']['rows']
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError('invalid manifest row count')
    boundaries = [item['row'] for item in result['boundaries']]
    if boundaries != sorted(set(boundaries)) or any(not isinstance(row, int) or not 0 < row < rows for row in boundaries):
        raise ValueError('invalid source boundaries')
    sources = result.get('sources', [])
    if sources:
        cursor = 0
        for source in sources:
            if source['start_row'] != cursor or source['rows'] <= 0:
                raise ValueError('source ranges are not contiguous')
            cursor += source['rows']
            if source['end_row_exclusive'] != cursor:
                raise ValueError('invalid source end row')
        if cursor != rows or boundaries != [source['start_row'] for source in sources[1:]]:
            raise ValueError('source ranges and boundaries disagree')
    return result


def describe_concatenation(data, source_paths, names=None, verify_order=True):
    """Describe an existing CSV; never rewrite it or invent missing sample provenance."""
    data = Path(data)
    frame = pd.read_csv(data)
    names = names or [Path(path).stem for path in source_paths]
    if len(names) != len(source_paths) or not source_paths:
        raise ValueError('each source needs a name')
    sources = []
    cursor = 0
    for path, name in zip(source_paths, names):
        path = Path(path)
        count = 0
        with pd.read_csv(path, chunksize=50000) as reader:
            for chunk in reader:
                if chunk.columns.tolist() != frame.columns.tolist():
                    raise ValueError('source schema/order differs from concatenated CSV')
                if verify_order:
                    actual = frame.iloc[cursor + count:cursor + count + len(chunk)].reset_index(drop=True)
                    try:
                        pd.testing.assert_frame_equal(actual, chunk.reset_index(drop=True),
                                                      check_dtype=False, check_exact=False,
                                                      rtol=1e-12, atol=1e-12)
                    except AssertionError as error:
                        raise ValueError('CSV is not the declared ordered concatenation') from error
                count += len(chunk)
        if count <= 0:
            raise ValueError('source CSV is empty')
        item = dict(name=name, path=str(path.resolve()), sha256=sha256_file(path),
                    rows=count, start_row=cursor, end_row_exclusive=cursor + count)
        sidecar = manifest_path(path)
        if sidecar.exists():
            provenance = read_manifest(path, sidecar)
            if provenance['data']['rows'] != count or provenance['data']['columns'] != frame.columns.tolist():
                raise ValueError('source manifest row count or schema does not match')
            if provenance['feature_columns'] != select_features(frame).columns.tolist():
                raise ValueError('source manifest feature order differs from combined data')
            item['provenance'] = provenance
        else:
            item['provenance'] = {'status': 'sampling indices and seed not recorded in source artifact'}
        sources.append(item)
        cursor += count
    if cursor != len(frame):
        raise ValueError('source lengths do not sum to combined CSV length')
    return dict(schema_version=1,
                data=dict(path=str(data.resolve()), sha256=sha256_file(data),
                          rows=len(frame), columns=frame.columns.tolist()),
                feature_columns=select_features(frame).columns.tolist(),
                excluded_columns=[c for c in EXCLUDED_COLUMNS if c in frame],
                sources=sources,
                boundaries=[dict(row=b['start_row'], kind='source_change',
                                 **{'from': a['name'], 'to': b['name']}) for a, b in zip(sources, sources[1:])],
                order_verified=verify_order)


def write_manifest(data, metadata, path=None):
    path = Path(path) if path else manifest_path(data)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    return path
