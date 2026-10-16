"""Export bounded, verified A1 evidence; never synthesize missing measurements."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import shutil
from A1.protocol import ROOT, CONFIG, load_protocol


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def expected_source():
    names = ['__init__.py', 'protocol.py', 'models.py', 'data.py', 'metrics.py', 'engine.py', 'run_suite.py', 'visuals.py']
    paths = [ROOT / 'src/A1' / name for name in names] + [ROOT / 'src/common/runtime.py']
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def read_run(run, experiment, seed, protocol, *, source_files=None):
    path = run / 'metadata.json'
    if not path.exists():
        return None, 'missing metadata'
    metadata = json.loads(path.read_text())
    signature = metadata['signature']
    if (signature['protocol'] != protocol or signature['experiment'] != experiment or signature['seed'] != seed
            or signature['source'] != (expected_source() if source_files is None else source_files) or signature['config_sha256'] != digest(CONFIG)):
        raise ValueError('Protocol/source/config/seed mismatch')
    from hashlib import sha256
    signature_hash = sha256(json.dumps(signature, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if metadata['signature_sha256'] != signature_hash:
        raise ValueError('Signature hash mismatch')
    split = json.loads((run / 'split.json').read_text())
    split_hash = sha256(json.dumps(split, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if split_hash != signature['data']['split_sha256'] or split['train_count'] != 45000 or split['val_count'] != 5000:
        raise ValueError('Split artifact mismatch')
    with (run / 'history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    if len(history) != metadata['completed_epochs']:
        raise ValueError('History/metadata mismatch; resume to recover committed epoch artifacts')
    if metadata['status'] != 'trained' or len(history) != protocol['epochs']:
        return None, f'incomplete training ({len(history)}/{protocol["epochs"]})'
    best = metadata['best_validation']
    selected = history[best['epoch'] - 1]
    if float(selected['val_accuracy']) != best['accuracy'] or float(selected['val_ce']) != best['ce']:
        raise ValueError('Selected checkpoint/history mismatch')
    winner = max(history, key=lambda row: (float(row['val_accuracy']), -float(row['val_ce']), -int(row['epoch'])))
    if int(winner['epoch']) != best['epoch']:
        raise ValueError('Checkpoint was not selected by the declared validation rule')
    row = {'experiment': experiment['id'], 'seed': seed, 'selected_epoch': best['epoch'],
           'val_accuracy': best['accuracy'], 'val_macro_f1': best['macro_f1'],
           'test_accuracy': None, 'test_macro_f1': None,
           'train_seconds': metadata['train_seconds'], 'peak_gpu_memory_bytes': metadata['peak_gpu_memory_bytes'],
           'total_params': metadata['total_params'], 'trainable_params': metadata['trainable_params'],
           'signature_sha256': signature_hash, 'best_state_sha256': best['state_sha256'],
           'data': signature['data'], 'environment': signature['environment'], 'num_workers': signature['num_workers']}
    evaluation_path = run / 'evaluation_test.json'
    reason = 'test not evaluated'
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        identity = evaluation['identity']
        if (identity['signature_sha256'] != signature_hash or evaluation['best_state_sha256'] != best['state_sha256']
                or evaluation['selected_epoch'] != best['epoch'] or identity['test_data']['count'] != 10000
                or identity['test_data']['classes'] != split['classes']):
            raise ValueError('Test/checkpoint/data provenance mismatch')
        if (run / 'best.pt').exists() and digest(run / 'best.pt') != identity['best_file_sha256']:
            raise ValueError('Test refers to another checkpoint file')
        for name, expected in evaluation['artifacts'].items():
            if digest(run / name) != expected:
                raise ValueError(f'Altered evaluation artifact: {name}')
        if sum(sum(r) for r in evaluation['confusion']) != 10000:
            raise ValueError('Test confusion does not cover 10000 images')
        row.update(test_accuracy=evaluation['accuracy'], test_macro_f1=evaluation['macro_f1'],
                   test_data=identity['test_data'], best_file_sha256=identity['best_file_sha256'])
        reason = 'complete'
    for key in ['val_accuracy', 'val_macro_f1', 'test_accuracy', 'test_macro_f1', 'train_seconds', 'peak_gpu_memory_bytes']:
        value = row[key]
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f'Invalid metric {key}')
    return row, reason


def aggregate(rows, experiment):
    selected = [row for row in rows if row['experiment'] == experiment['id']]
    result = {'experiment': experiment['id'], 'train_seeds': [r['seed'] for r in selected],
              'test_seeds': [r['seed'] for r in selected if r['test_accuracy'] is not None]}
    for key in ['val_accuracy', 'val_macro_f1', 'test_accuracy', 'test_macro_f1', 'train_seconds',
                'peak_gpu_memory_bytes', 'total_params', 'trainable_params']:
        values = [r[key] for r in selected if r[key] is not None]
        result[key] = {'mean': statistics.mean(values) if values else None,
                       'sd': statistics.stdev(values) if len(values) > 1 else None, 'n': len(values)}
    return result


def export(output_root, destination, *, source_files=None):
    protocol = load_protocol()
    rows, coverage = [], []
    for experiment in protocol['experiments']:
        for seed in protocol['seeds']:
            run = output_root / experiment['id'] / f'seed_{seed}'
            try:
                row, status = read_run(run, experiment, seed, protocol, source_files=source_files)
            except (KeyError, ValueError, OSError, IndexError, TypeError) as exc:
                row, status = None, f'rejected: {exc}'
            if row:
                # Reject cross-run split / test dataset differences, rather than
                # averaging incomparable data. Runtime grouping is also explicit.
                if rows and row['data'] != rows[0]['data']:
                    row, status = None, 'rejected: cross-run development data mismatch'
                elif row['test_accuracy'] is not None:
                    prior = next((r for r in rows if r['test_accuracy'] is not None), None)
                    if prior and row['test_data'] != prior['test_data']:
                        row, status = None, 'rejected: cross-run test data mismatch'
            if row:
                rows.append(row)
                evidence = destination / 'evidence' / experiment['id'] / f'seed_{seed}'
                current_files = {'split.json'}
                for name in ['metadata.json', 'history.csv', 'evaluation_test.json', 'per_class_test.csv',
                             'confusion_test.csv', 'predictions_test.csv']:
                    if (run / name).exists():
                        atomic_copy(run / name, evidence / name)
                        current_files.add(name)
                # split.json contains indices; preserve it once per experiment
                # seed evidence here for independently auditable data provenance.
                atomic_copy(run / 'split.json', evidence / 'split.json')
                for stale in evidence.iterdir():
                    if stale.is_file() and stale.name not in current_files:
                        stale.unlink()
                coverage.append({'experiment': experiment['id'], 'seed': seed, 'status': status,
                                 'evidence_files': {p.name: digest(p) for p in evidence.iterdir() if p.is_file()}})
            else:
                coverage.append({'experiment': experiment['id'], 'seed': seed, 'status': status})
    summary = [aggregate(rows, e) for e in protocol['experiments']]
    environments = {json.dumps({'environment': r['environment'], 'workers': r['num_workers']}, sort_keys=True) for r in rows}
    # Time summaries are meaningful only inside an identical environment.
    for group in summary:
        group_env = {json.dumps({'environment': r['environment'], 'workers': r['num_workers']}, sort_keys=True)
                     for r in rows if r['experiment'] == group['experiment']}
        group['timing_comparable'] = len(group_env) <= 1
        if len(group_env) > 1:
            group['train_seconds'] = {'mean': None, 'sd': None, 'n': 0}
            group['peak_gpu_memory_bytes'] = {'mean': None, 'sd': None, 'n': 0}
    manifest = {'protocol': protocol, 'config_sha256': digest(CONFIG), 'source': expected_source(),
                'expected_runs': 24, 'trained_runs': len(rows),
                'tested_runs': sum(r['test_accuracy'] is not None for r in rows),
                'coverage': coverage, 'cross_configuration_timing_comparable': len(environments) <= 1,
                'state': 'no measurements' if not rows else 'partial' if len(rows) < 24 or any(r['test_accuracy'] is None for r in rows) else 'complete',
                'note': 'Means use only seeds listed; SD undefined for n<2. Figures use the smallest available seed, never the best test seed.'}
    write_json(destination / 'manifest.json', manifest)
    write_json(destination / 'summary.json', {'groups': summary, 'runs': rows})
    fields = ['experiment', 'train_seeds', 'test_seeds'] + [f'{key}_{part}' for key in
              ['val_accuracy', 'val_macro_f1', 'test_accuracy', 'test_macro_f1', 'train_seconds',
               'peak_gpu_memory_bytes', 'total_params', 'trainable_params'] for part in ['mean', 'sd', 'n']]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for group in summary:
        flattened = {key: group[key] for key in ['experiment', 'train_seeds', 'test_seeds']}
        flattened.update({f'{key}_{part}': group[key][part] for key in group if isinstance(group[key], dict)
                          for part in ['mean', 'sd', 'n']})
        writer.writerow(flattened)
    atomic_text(destination / 'summary.csv', buffer.getvalue())
    def formatted(metric, percent=False):
        if metric['mean'] is None:
            return '--'
        scale = 100 if percent else 1
        value = f"{metric['mean'] * scale:.2f}"
        return value + (f" $\\pm$ {metric['sd'] * scale:.2f}" if metric['sd'] is not None else '')
    lines = ['\\begin{tabular}{lrrrrr}', '\\toprule',
             'Cấu hình & Val (\\%) & Test (\\%) & F1 test & Train (s) & Trainable \\\\', '\\midrule']
    for group in summary:
        lines.append(group['experiment'].replace('_', r'\_') + ' & ' + ' & '.join([
            formatted(group['val_accuracy'], True), formatted(group['test_accuracy'], True),
            formatted(group['test_macro_f1']), formatted(group['train_seconds']),
            '--' if group['trainable_params']['mean'] is None else f"{group['trainable_params']['mean']:,.0f}"]) + r' \\')
    lines += ['\\bottomrule', '\\end{tabular}',
              f'\\par\\smallskip{{\\small {manifest["trained_runs"]}/24 run train hoàn tất; {manifest["tested_runs"]}/24 run test. --: chưa đo. SD chỉ khi ít nhất hai seed.}}']
    atomic_text(destination / 'comparison_table.tex', '\n'.join(lines) + '\n')
    figure_records = []
    for experiment in protocol['experiments']:
        selected = sorted((r for r in rows if r['experiment'] == experiment['id']), key=lambda r: r['seed'])
        if not selected:
            continue
        # Smallest trained seed: if its test is absent, do not pick a different
        # test seed to quietly improve coverage or select a flattering gallery.
        row = selected[0]
        run = output_root / experiment['id'] / f'seed_{row["seed"]}'
        for name in ['curves.png', 'confusion_test.png', 'errors_test.png']:
            if name != 'curves.png' and row['test_accuracy'] is None:
                continue
            if (run / name).exists():
                destination_name = f'{experiment["id"]}_{name}'
                atomic_copy(run / name, destination / 'figures' / destination_name)
                figure_records.append({'file': destination_name, 'seed': row['seed'], 'sha256': digest(run / name)})
    manifest['figures'] = figure_records
    write_json(destination / 'manifest.json', manifest)
    # Remove stale generated figures/evidence rejected by this export so old
    # measurements cannot survive as apparently current report content.
    allowed_figures = {r['file'] for r in figure_records}
    for path in (destination / 'figures').glob('*.png'):
        if path.name not in allowed_figures:
            path.unlink()
    accepted = {(r['experiment'], f'seed_{r["seed"]}') for r in rows}
    for path in (destination / 'evidence').glob('*/seed_*'):
        if (path.parent.name, path.name) not in accepted:
            shutil.rmtree(path)
    print(f'Exported {len(rows)}/24 trained, {manifest["tested_runs"]}/24 tested to {destination}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/assignment1')
    parser.add_argument('--destination', type=Path, default=ROOT / 'reports/assignment1/generated')
    parser.add_argument('--archive', action='store_true', help='Read the verified 1250e56 training source; never enables resume')
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination != ROOT / 'reports/assignment1/generated' and ROOT / 'reports/assignment1' not in destination.parents:
        parser.error('A1 export destination must remain under reports/assignment1/')
    from common.archive import a1_source
    export(args.output_root, destination, source_files=a1_source() if args.archive else None)


if __name__ == '__main__':
    main()
