"""Configurable full-stream runner; importing this module has no side effects."""
import argparse
import copy
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

import mpcawin
import kswin_compat
from mpcawin import MPCAWIN
from kswin_compat import KSWIN, RiverKSWINAdapter, positive_integer
from dataset_manifest import read_manifest, select_features, sha256_file

PAPER = dict(name='paper', description='author-code baseline on a reconstructed subset',
             batch_size=2000, min_window_size=10, max_window_size=10, coef_std=2,
             detect_strategy='detector', backend='legacy', kswin_alpha=.009,
             kswin_window_size=20, kswin_stat_size=5, seed=20, scaling='reference',
             rounding='legacy', pca_solver='auto', alert_window_size=20,
             alert_alpha=.5, remainder='error', matching_horizon_batches=20)


def resolve_config(config, directory=Path('.'), seen=()):
    config = copy.deepcopy(config)
    parent = config.pop('extends', None)
    result = PAPER.copy()
    if parent:
        path = (directory / parent).resolve()
        if path in seen:
            raise ValueError('config inheritance cycle')
        result = resolve_config(json.loads(path.read_text()), path.parent, seen + (path,))
    unknown = set(config) - set(PAPER)
    if unknown:
        raise ValueError(f'unknown config fields: {sorted(unknown)}')
    result.update(config)
    positive_integer(result['batch_size'], 'batch_size')
    positive_integer(result['matching_horizon_batches'], 'matching_horizon_batches')
    if result['remainder'] not in ('error', 'drop', 'process'):
        raise ValueError('remainder must be error, drop or process')
    if result['backend'] not in ('legacy', 'river'):
        raise ValueError('backend must be legacy or river')
    if not result['name'] or Path(result['name']).name != result['name'] or result['name'] in ('.', '..'):
        raise ValueError('configuration name must be a safe directory name')
    return result


def load_configs(path):
    if path is None:
        return [PAPER.copy()]
    path = Path(path)
    contents = json.loads(path.read_text())
    items = contents['configurations'] if 'configurations' in contents else [contents]
    configs = [resolve_config(item, path.parent) for item in items]
    if len({config['name'] for config in configs}) != len(configs):
        raise ValueError('configuration names must be unique')
    return configs


def make_detector(config):
    backend = KSWIN if config['backend'] == 'legacy' else RiverKSWINAdapter
    confirmation = backend(alpha=config['kswin_alpha'], window_size=config['kswin_window_size'],
                           stat_size=config['kswin_stat_size'], random_state=config['seed'])
    return MPCAWIN(min_window_size=config['min_window_size'], max_window_size=config['max_window_size'],
                   coef_std=config['coef_std'], detect_strategy=config['detect_strategy'],
                   alert_window_size=config['alert_window_size'], alpha=config['alert_alpha'],
                   detector=confirmation, random_seed=config['seed'], scaling=config['scaling'],
                   rounding=config['rounding'], pca_solver=config['pca_solver'])


def evaluate_events(events, boundaries, batch_size, horizon, processed_rows=None):
    """Match at most one event per source change within an explicit batch horizon."""
    remaining = sorted(events)
    matches = []
    for index, boundary in enumerate(boundaries):
        start_batch = boundary // batch_size
        next_batch = boundaries[index+1] // batch_size if index+1 < len(boundaries) else math.inf
        candidates = [event for event in remaining
                      if start_batch <= event <= start_batch + horizon and event < next_batch]
        event = candidates[0] if candidates else None
        if event is not None:
            remaining.remove(event)
        matches.append(dict(boundary_row=boundary, boundary_batch_id0=start_batch,
                            detection_batch_id0=event,
                            index_delay_batches=None if event is None else event-start_batch,
                            consumed_new_concept_batches=None if event is None else event-start_batch+1,
                            consumed_new_concept_records=None if event is None else
                            min((event+1)*batch_size, processed_rows if processed_rows is not None else math.inf)-boundary,
                            missed=event is None))
    return dict(matches=matches, unmatched_detection_batches=remaining,
                detections_before_first_boundary=[e for e in events if boundaries and e*batch_size < boundaries[0]],
                matching_horizon_batches=horizon)


def run_stream(features, config, boundaries=(), sources=()):
    size = config['batch_size']
    remainder = len(features) % size
    if remainder and config['remainder'] == 'error':
        raise ValueError(f'{remainder} remaining rows; choose drop or process explicitly')
    count = len(features)//size + int(bool(remainder) and config['remainder']=='process')
    model = make_detector(config)
    rows = []
    with threadpool_limits(limits=1):
        for batch_id in range(count):
            start, end = batch_id*size, min((batch_id+1)*size,len(features))
            model.add_element_and_detected_change(features.iloc[start:end])
            result = model.last_result
            ks = result['ks'] or {}
            containing = [s['name'] for s in sources if s['start_row'] < end and s['end_row_exclusive'] > start]
            source = containing[0] if len(containing)==1 else 'Mixed' if containing else None
            row = {key:value for key,value in result.items() if key!='ks'}
            row.update(batch_id=batch_id, batch_number1=batch_id+1, start_row=start,
                       end_row_exclusive=end, source=source,
                       ks_tested=ks.get('tested',False), ks_statistic=ks.get('statistic'),
                       ks_p_value=ks.get('p_value'), ks_window_length=ks.get('window_length'),
                       ks_reference=json.dumps(ks.get('reference')) if ks.get('reference') is not None else None,
                       ks_recent=json.dumps(ks.get('recent')) if ks.get('recent') is not None else None)
            rows.append(row)
    table = pd.DataFrame(rows)
    events = table.loc[table.drift_detected,'batch_id'].tolist() if len(table) else []
    processed_rows = count*size if remainder and config['remainder']=='drop' else len(features)
    summary = dict(configuration=config, rows=len(features), processed_rows=processed_rows,
                   dropped_rows=remainder if config['remainder']=='drop' else 0,
                   detected_batches=events, evaluated_batches=int(table.alert.notna().sum()) if len(table) else 0,
                   initialization_batches=int((table.phase=='initialize').sum()) if len(table) else 0,
                   warmup_batches=int((table.phase=='warmup').sum()) if len(table) else 0,
                   concept_reset_count=len(events),
                   alert_count=int(table.alert.sum()) if len(table) else 0,
                   evaluation=evaluate_events(events,list(boundaries),size,config['matching_horizon_batches'],processed_rows))
    return table, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--boundary-row', type=int, help='explicit override; may not contradict a supplied manifest')
    args = parser.parse_args(argv)
    sidecar = args.manifest or args.data.with_suffix('.manifest.json')
    if sidecar.exists():
        manifest = read_manifest(args.data,sidecar)
        boundaries = [item['row'] for item in manifest['boundaries']]
        if args.boundary_row is not None and boundaries != [args.boundary_row]:
            parser.error('boundary override contradicts manifest')
    elif args.manifest is not None or args.boundary_row is None:
        parser.error('dataset manifest is required, or provide an explicit --boundary-row')
    else:
        manifest = None
        boundaries = [args.boundary_row]
    frame = pd.read_csv(args.data)
    if any(not 0 < boundary < len(frame) for boundary in boundaries):
        parser.error('source boundary is outside the dataset')
    if manifest and (len(frame)!=manifest['data']['rows'] or frame.columns.tolist()!=manifest['data']['columns']):
        raise ValueError('dataset row count or schema disagrees with manifest')
    features = select_features(frame,manifest['feature_columns'] if manifest else None)
    provenance = runtime_provenance(args.data, sidecar if manifest else None, features)
    for config in load_configs(args.config):
        table, summary = run_stream(features,config,boundaries,manifest['sources'] if manifest else ())
        output = args.output_dir / config['name']
        output.mkdir(parents=True,exist_ok=True)
        table.to_csv(output/'batches.csv',index=False)
        summary['provenance'] = provenance
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f"{config['name']}: detections={summary['detected_batches']}; "
              f"misses={sum(item['missed'] for item in summary['evaluation']['matches'])}; output={output}",flush=True)
    return 0


def runtime_provenance(data, sidecar, features):
    return dict(data=str(Path(data).resolve()), data_sha256=sha256_file(data),
                manifest=str(Path(sidecar).resolve()) if sidecar else None,
                manifest_sha256=sha256_file(sidecar) if sidecar else None,
                features=features.columns.tolist(), python=sys.version.split()[0],
                interpreter=sys.executable, mpcawin_source=mpcawin.__file__,
                kswin_source=kswin_compat.__file__,
                versions={name:version(name) for name in
                          ['numpy','pandas','scipy','scikit-learn','river','mpcawin']})
