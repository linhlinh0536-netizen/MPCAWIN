"""Full-stream regression against archived outputs and multi-seed experiments."""
import argparse
import json
from pathlib import Path

import pandas as pd
from dataset_manifest import read_manifest, select_features, sha256_file
from mpcawin_run import load_configs, run_stream, runtime_provenance

BASE = Path(__file__).resolve().parents[1]


def compare_results(old, new, mapping):
    expected = old[list(mapping)].rename(columns=mapping).reset_index(drop=True)
    actual = new[list(mapping.values())].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False,
                                  check_exact=False, rtol=1e-12, atol=1e-12)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10))+[20])
    parser.add_argument('--output-dir', type=Path, default=BASE/'results'/'validation')
    args = parser.parse_args(argv)
    metadata = read_manifest(args.data, args.manifest)
    frame = pd.read_csv(args.data)
    if len(frame) != metadata['data']['rows'] or frame.columns.tolist() != metadata['data']['columns']:
        raise ValueError('dataset shape/schema mismatch')
    features = select_features(frame, metadata['feature_columns'])
    boundaries = [item['row'] for item in metadata['boundaries']]
    paper = load_configs(BASE/'configs'/'paper.json')[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    current, _ = run_stream(features, paper, boundaries, metadata['sources'])
    archived = pd.read_csv(BASE/'diagnostics'/'paper_results'/'author_c2_reference_scaler.csv')
    compare_results(archived, current, dict(batch_id='batch_id', angle='angle',
        threshold_used='threshold_used', alert='alert', p_value='ks_p_value',
        statistic='ks_statistic', drift='drift_detected'))
    current_c3, _ = run_stream(features, dict(paper, coef_std=3), boundaries, metadata['sources'])
    archived_c3 = pd.read_csv(BASE/'baselines'/'before_repair'/'mpcawin_results.csv')
    compare_results(archived_c3, current_c3, {key:key for key in
        ['batch_id','angle','threshold_used','alert','noise_angle','ks_p_value','ks_statistic','drift_detected']})
    print('PASS: full 160-batch c=2 and c=3 regression against archived outputs', flush=True)

    selected = [config for config in load_configs(BASE/'configs'/'experiments.json')
                if config['name'] in ('legacy_60_20','river_60_20','alert_rate_20')]
    records = []
    for base_config in selected:
        for seed in args.seeds:
            config = dict(base_config, seed=seed)
            table, summary = run_stream(features, config, boundaries, metadata['sources'])
            table.to_csv(args.output_dir/f"{config['name']}_seed{seed}.csv", index=False)
            evaluation = summary['evaluation']
            event_rows = table.loc[table.drift_detected]
            if config['detect_strategy'] == 'detector' and len(event_rows):
                if not event_rows.ks_tested.all() or not (event_rows.ks_p_value <= config['kswin_alpha']).all():
                    raise AssertionError('KS event evidence was lost during reset')
            records.append(dict(configuration=config['name'], seed=seed,
                events=summary['detected_batches'],
                misses=sum(item['missed'] for item in evaluation['matches']),
                pre_boundary_events=len(evaluation['detections_before_first_boundary']),
                unmatched_events=evaluation['unmatched_detection_batches'],
                index_delays=[item['index_delay_batches'] for item in evaluation['matches']]))
        group = records[-len(args.seeds):]
        print(f"{base_config['name']}: {sum(r['misses'] for r in group)}/{len(group)} missed runs; "
              f"pre-boundary events={sum(r['pre_boundary_events'] for r in group)}", flush=True)
    result = dict(data_sha256=sha256_file(args.data), rows=len(features),
        provenance=runtime_provenance(args.data,args.manifest or args.data.with_suffix('.manifest.json'),features),
        features=features.columns.tolist(), seeds=args.seeds,
        full_stream_parity=dict(c2=True,c3=True,compared_batches=len(current)),
        configurations=selected, runs=records,
        scope='Local reconstructed subset; source boundaries are proxies, unmatched events are not assumed false positives')
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0


if __name__ == '__main__':
    main()
