"""Export measured results to LaTeX; never substitute estimates for missing runs."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
import torch
from common.config import load_experiments
from common.factory import build_model, parameter_count
from common.runtime import atomic_json, atomic_csv, json_hash


def collect_runs(root: Path) -> list[dict]:
    records = []
    for path in sorted(Path(root).glob('*/seed_*/metrics.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('status') != 'trained':
            continue
        if (path.parent / '.train.lock').exists():
            raise ValueError(f'Run is still locked: {path.parent}')
        records.append(dict(record, directory=path.parent))
    # Check this before reading detailed metrics: mixing split definitions is a
    # protocol error even when a run's other artifacts are incomplete.
    if len({r['data_fingerprint'] for r in records}) > 1:
        raise ValueError('Cannot compare runs with different data/split fingerprints.')
    expected = {c.id: c.to_dict() for c in load_experiments()}
    seen, source_hashes, precision_settings = set(), set(), set()
    for r in records:
        identifier, seed = r['experiment']['id'], r['seed']
        if identifier not in expected or r['experiment'] != expected[identifier]:
            raise ValueError(f'{identifier}: nonstandard config. Keep trial runs in a separate output root.')
        if (identifier, seed) in seen:
            raise ValueError(f'Duplicate run: {identifier}, seed {seed}')
        seen.add((identifier, seed))
        if r['completed_epochs'] != expected[identifier]['epochs']:
            raise ValueError(f'{identifier}: incomplete epoch budget.')
        for key in ['best_validation_accuracy', 'best_validation_ce', 'median_train_seconds']:
            if not math.isfinite(r[key]):
                raise ValueError(f'Non-finite {key}: {identifier}')
        if not 0 <= r['best_validation_accuracy'] <= 1:
            raise ValueError('Accuracy must be stored as a fraction in [0,1].')
        manifest = json.loads((r['directory'] / 'run.json').read_text(encoding='utf-8'))
        if manifest['signature'] != r['signature']:
            raise ValueError('Run manifest and summary signatures differ.')
        source_hashes.add(manifest['source_fingerprint'])
        precision_settings.add((manifest['dtype'], manifest['deterministic']))
        evaluation = r['directory'] / 'evaluation_test.json'
        r['test'] = None
        if evaluation.exists():
            e = json.loads(evaluation.read_text(encoding='utf-8'))
            if e['signature'] != r['signature'] or e['model_sha256'] != r['best_model_sha256']:
                raise ValueError(f'{identifier}: test evaluation is from a different checkpoint.')
            if e['split'] != 'test' or e['seed'] != seed:
                raise ValueError(f'{identifier}: wrong test split or seed.')
            for key in ['accuracy', 'macro_f1', 'inference_images_per_second']:
                if not math.isfinite(e[key]):
                    raise ValueError(f'Non-finite test metric: {key}')
            if not all(0 <= e[k] <= 1 for k in ['accuracy', 'macro_f1']):
                raise ValueError('Test metrics must be fractions in [0,1].')
            r['test'] = e
    if len(source_hashes) > 1:
        raise ValueError('Model/training source changed between runs. Do not pool different code revisions.')
    if len(precision_settings) > 1:
        raise ValueError('Precision/determinism settings differ between runs.')
    if len({r['test']['dataset_sha256'] for r in records if r['test']}) > 1:
        raise ValueError('Test dataset fingerprints differ between runs.')
    return records


def mean_or_none(values):
    return statistics.mean(values) if values else None


def latex_escape(text: str) -> str:
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c, c) for c in str(text))


def formatted(value, scale=1., digits=2):
    return '--' if value is None else f'{value*scale:.{digits}f}'


def table(rows: list[dict], target: Path):
    text = [r'\begin{tabularx}{\textwidth}{@{}Xrrrr@{}}', r'\toprule',
            r'Mô hình & Tham số & Val (\%) & Test (\%) & Giây/epoch \\', r'\midrule']
    for r in rows:
        test = formatted(r['test_accuracy_mean'], 100)
        if r['test_accuracy_std'] is not None:
            test += r' $\pm$ ' + formatted(r['test_accuracy_std'], 100)
        text.append(f"{latex_escape(r['label'])} & {r['parameters']:,} & "
                    f"{formatted(r['validation_accuracy_mean'],100)} & {test} & "
                    f"{formatted(r['median_train_seconds_mean'])}" + r' \\')
    text.extend([r'\bottomrule', r'\end{tabularx}'])
    target.write_text('\n'.join(text) + '\n', encoding='utf-8')


def export_figures(records: list[dict], directory: Path) -> dict:
    """Copy figures from the lowest seed for each configuration, not best test."""
    figures = directory / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    # This directory is generated-only. Remove stale figures when inputs change.
    for old in figures.glob('*.png'):
        old.unlink()
    selected = {}
    for r in records:
        key = r['experiment']['id']
        if key not in selected or r['seed'] < selected[key]['seed']:
            selected[key] = r
    provenance = {}
    for key, r in selected.items():
        for source_name, suffix in [('ce', 'ce'), ('accuracy', 'accuracy'),
                                    ('confusion_test', 'confusion'), ('mistakes_test', 'mistakes')]:
            source = r['directory'] / 'plots' / f'{source_name}.png'
            if source.is_file():
                name = f'{key}_{suffix}.png'
                shutil.copy2(source, figures / name)
                provenance[name] = dict(experiment=key, seed=r['seed'], signature=r['signature'])
        source = r['directory'] / 'diagnostics' / 'temporal_gradients.png'
        summary = source.with_name('summary.json')
        if source.is_file() and summary.is_file():
            diagnostic = json.loads(summary.read_text(encoding='utf-8'))
            if diagnostic['signature'] != r['signature'] or diagnostic['model_sha256'] != r['best_model_sha256']:
                raise ValueError('Stale gradient diagnostic.')
            name = f'{key}_gradients.png'
            shutil.copy2(source, figures / name)
            provenance[name] = dict(experiment=key, seed=r['seed'], signature=r['signature'])
    # Learning-curve comparisons use the same, smallest shared seed in each group.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group, ids in {
        'e1': ['e1_softmax', 'e1_mlp', 'e1_cnn'],
        'e2': [c.id for c in load_experiments() if c.exercise == 'E2'],
        'e3': ['e1_mlp', 'e1_cnn', 'e3_lstm_rows', 'e3_gru_rows', 'e3_lstm_patches', 'e3_gru_patches'],
        'improvement': ['e3_lstm_rows', 'e3_lstm_rows_orth_reg', 'e3_lstm_rows_chrono',
                        'e3_lstm_rows_mean', 'e3_lstm_rows_combined'],
    }.items():
        by_id = {key: [r for r in records if r['experiment']['id'] == key] for key in ids}
        if not all(by_id.values()):
            continue
        common = set.intersection(*[{r['seed'] for r in rs} for rs in by_id.values()])
        if not common:
            continue
        seed = min(common)
        for metric, ylabel in [('ce', 'Validation cross-entropy'), ('accuracy', 'Validation accuracy')]:
            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            for key in ids:
                run = next(r for r in by_id[key] if r['seed'] == seed)
                with (run['directory'] / 'history.csv').open(encoding='utf-8') as f:
                    history = list(csv.DictReader(f))
                ax.plot([int(h['epoch']) for h in history],
                        [float(h['validation_' + metric]) for h in history], label=run['experiment']['label'])
            ax.set(xlabel='Epoch', ylabel=ylabel, title=f'Seed {seed}')
            ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
            name = f'{group}_comparison_{metric}.png'
            fig.savefig(figures / name, dpi=150); plt.close(fig)
            provenance[name] = dict(experiments=ids, seed=seed)
    return provenance


def export_report(root: Path, report_root: Path) -> list[dict]:
    records = collect_runs(root)
    generated = Path(report_root) / 'generated'; generated.mkdir(parents=True, exist_ok=True)
    rows = []
    # Counting parameters constructs models but never reads data or runs forward.
    with torch.random.fork_rng(devices=[]):
        for config in load_experiments():
            own = [r for r in records if r['experiment']['id'] == config.id]
            tested = [r for r in own if r['test'] is not None]
            params = parameter_count(build_model(config))
            if any(r['n_parameters'] != params for r in own):
                raise ValueError(f'{config.id}: parameter count differs from the current architecture.')
            test_accuracy = [r['test']['accuracy'] for r in tested]
            timing_environments = {json_hash({k: r['environment'].get(k) for k in
                ['python', 'torch', 'cuda_runtime', 'cudnn', 'device', 'device_name', 'torch_num_threads', 'num_workers']}) for r in own}
            # Accuracy can be summarized across devices, but speed should not be.
            timing = mean_or_none([r['median_train_seconds'] for r in own]) if len(timing_environments) <= 1 else None
            rows.append(dict(id=config.id, label=config.label, exercise=config.exercise, group=config.group,
                parameters=params, n_train_runs=len(own), n_test_runs=len(tested),
                train_seeds=[r['seed'] for r in own], test_seeds=[r['seed'] for r in tested],
                validation_accuracy_mean=mean_or_none([r['best_validation_accuracy'] for r in own]),
                test_accuracy_mean=mean_or_none(test_accuracy),
                test_accuracy_std=statistics.stdev(test_accuracy) if len(test_accuracy) >= 2 else None,
                test_macro_f1_mean=mean_or_none([r['test']['macro_f1'] for r in tested]),
                median_train_seconds_mean=timing, mixed_timing_environments=len(timing_environments) > 1))
    for exercise in ['E1', 'E2', 'E3']:
        chosen = [r for r in rows if r['exercise'] == exercise and r['group'] == 'required']
        if exercise == 'E3':
            chosen = [r for r in rows if r['id'] in ['e1_mlp', 'e1_cnn']] + chosen
        table(chosen, generated / f'{exercise.lower()}_table.tex')
    table([r for r in rows if r['id'] == 'e3_lstm_rows' or r['group'] == 'improvement'], generated / 'improvement_table.tex')
    train_count, test_count = len(records), sum(r['test'] is not None for r in records)
    status = (r'\textbf{Chưa huấn luyện.} Các bảng chỉ có số tham số đếm từ kiến trúc; dấu -- nghĩa là chưa đo.'
              if not train_count else
              f'Đã tổng hợp {train_count} lượt huấn luyện và {test_count} lượt đánh giá test. '
              'Các ô trống vẫn chưa có số liệu; xem số seed thực tế trong bảng phạm vi bên dưới.')
    (generated / 'status.tex').write_text(status + '\n', encoding='utf-8')
    coverage = [r'\begin{longtable}{@{}p{0.65\textwidth}rr@{}}', r'\toprule',
                r'Cấu hình & Seed train & Seed test \\ \midrule \endhead']
    for r in rows:
        coverage.append(f"{latex_escape(r['label'])} & {','.join(map(str,r['train_seeds'])) or '--'} & "
                        f"{','.join(map(str,r['test_seeds'])) or '--'}" + r' \\')
    coverage.extend([r'\bottomrule', r'\end{longtable}'])
    if not records:
        coverage = ['Chưa có seed nào hoàn tất. Nghiên cứu có 20 cấu hình; kế hoạch ba seed '
                    '42, 43, 44 tương ứng 60 lượt. Danh sách cấu hình nằm trong '
                    r'\texttt{configs/experiments.json}; đây không phải số lượt đã chạy.']
    (generated / 'coverage.tex').write_text('\n'.join(coverage)+'\n', encoding='utf-8')
    figures = export_figures(records, generated)
    atomic_json(generated / 'summary.json', rows)
    atomic_csv(generated / 'summary.csv', [{**r, 'train_seeds': ','.join(map(str,r['train_seeds'])),
                   'test_seeds': ','.join(map(str,r['test_seeds']))} for r in rows])
    atomic_json(generated / 'manifest.json', dict(train_runs=train_count, test_runs=test_count,
        runs=[dict(experiment=r['experiment']['id'], seed=r['seed'], signature=r['signature'],
                   best_model_sha256=r['best_model_sha256']) for r in records], figures=figures,
        figure_rule='Lowest seed per configuration; lowest shared seed for comparison curves. Never selected on test accuracy.',
        missing_measurements='null in JSON, empty CSV cells, -- in LaTeX',
        standard_deviation='Sample standard deviation across seeds; undefined for one seed, not zero.'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--report-root', type=Path, default=Path('reports'))
    args = parser.parse_args()
    rows = export_report(args.runs, args.report_root)
    print(f"{len(rows)} configurations; {sum(r['n_train_runs'] for r in rows)} completed runs.")


if __name__ == '__main__':
    main()
