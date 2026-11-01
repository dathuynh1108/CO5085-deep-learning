"""Build the course site from separate reports and verified measured summaries."""
from pathlib import Path
import argparse
import csv
import math
import statistics
import html
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def accuracy_text(mean, sd):
    if mean is None:
        return 'Chưa đo'
    if sd is None:
        return f'{mean*100:.2f}% · SD chưa xác định'
    return f'{mean*100:.2f} ± {sd*100:.2f}%'


A1_NAMES = {
    'resnet18_scratch': 'ResNet18 — scratch',
    'resnet18_head': 'ResNet18 — head-only',
    'resnet18_partial': 'ResNet18 — partial fine-tuning',
    'resnet18_full': 'ResNet18 — full fine-tuning',
    'swin_t_scratch': 'Swin-T — scratch',
    'swin_t_head': 'Swin-T — head-only',
    'swin_t_partial': 'Swin-T — partial fine-tuning',
    'swin_t_full': 'Swin-T — full fine-tuning',
}


def result_markup(exercises):
    rows = []
    for key in ['e1_softmax', 'e1_mlp', 'e1_cnn', 'e3_gru_rows']:
        row = next(r for r in exercises if r['id'] == key)
        mean, sd = row['test_accuracy_mean'], row['test_accuracy_std']
        value = accuracy_text(mean, sd)
        width = 0 if mean is None else mean*100
        rows.append(f'<div class="result-row"><span>{html.escape("GRU (hàng ảnh)" if key == "e3_gru_rows" else row["label"])}</span><div class="result-track"><i style="--score:{width:.3f}%"></i></div><strong>{value}</strong></div>')
    return ''.join(rows)


def a1_table_rows(assignment):
    a1 = []
    for row in assignment['groups']:
        mean, sd = row['test_accuracy']['mean'], row['test_accuracy']['sd']
        value = accuracy_text(mean, sd)
        f1 = row['test_macro_f1']['mean']
        a1.append(f'<tr><th scope="row">{html.escape(A1_NAMES[row["experiment"]])}</th><td>{value}</td><td>{"—" if f1 is None else f"{f1:.4f}"}</td></tr>')
    return ''.join(a1)


def a2_markup(report):
    evidence = report / 'notebook-evidence'
    with (evidence / 'results.csv').open(newline='') as stream:
        results = list(csv.DictReader(stream))
    variants = ('FULL', 'NO_BAS')
    variant_names = {'FULL': 'PIDNet-S (đủ loss)', 'NO_BAS': 'PIDNet-S (bỏ BAS-Loss)',
                     'PIDNet-S, published': 'PIDNet-S (paper)',
                     'Released author checkpoint': 'Checkpoint tác giả'}
    if len(results) != 6 or any(
        sorted(int(row['seed']) for row in results if row['variant'] == variant) != [42, 43, 44]
        for variant in variants
    ):
        raise ValueError('A2 requires FULL and NO_BAS test results for seeds 42, 43, and 44.')
    metrics = {}
    rows = []
    for variant in variants:
        group = [row for row in results if row['variant'] == variant]
        metrics[variant] = {}
        for key in ('test_miou', 'test_macro_dice', 'train_seconds'):
            values = [float(row[key]) for row in group]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'Nonfinite A2 metric: {variant}/{key}')
            metrics[variant][key] = (statistics.mean(values), statistics.stdev(values))
        values = [accuracy_text(*metrics[variant][key]).replace('.', ',')
                  for key in ('test_miou', 'test_macro_dice')]
        seconds, sd = metrics[variant]['train_seconds']
        values.append(f'{seconds / 60:.2f} ± {sd / 60:.2f} phút'.replace('.', ','))
        rows.append(f'<tr><th scope="row">{html.escape(variant_names[variant])}</th>' +
                    ''.join(f'<td>{value}</td>' for value in values) + '</tr>')
    paired = []
    for seed in (42, 43, 44):
        values = {row['variant']: float(row['test_miou']) for row in results if int(row['seed']) == seed}
        paired.append(values['FULL'] - values['NO_BAS'])
    delta = f'{100 * statistics.mean(paired):+.2f}'.replace('.', ',').replace('-', '−')
    delta_sd = f'{100 * statistics.stdev(paired):.2f}'.replace('.', ',')
    finding = (f'Bản đủ loss có mIoU cao hơn bản bỏ BAS-Loss trung bình {delta.lstrip("+")} ± {delta_sd} '
               'điểm phần trăm qua ba cặp seed. Kiến trúc và thiết lập train được giữ giống nhau.')
    with (evidence / 'paper_comparison.csv').open(newline='') as stream:
        comparison = list(csv.DictReader(stream))
    comparison_rows = []
    for row in comparison:
        miou = float(row['mIoU (%)'])
        fps = float(row['FPS']) if row['FPS'] else float('nan')
        values = [row['Training images'], f'{miou:.2f}%'.replace('.', ','),
                  row['GPU'], '—' if not math.isfinite(fps) else f'{fps:.1f}'.replace('.', ',')]
        comparison_rows.append(f'<tr><th scope="row">{html.escape(variant_names.get(row["Model"], row["Model"]))}</th>' +
                               ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>')
    return [('A2_STATUS', '6 lượt huấn luyện và 6 lượt đánh giá trên tập test CamVid.'),
            ('A2_RESULTS', ''.join(rows)), ('A2_FINDING', html.escape(finding)),
            ('A2_COMPARISON', ''.join(comparison_rows))]


def measured_copy(exercises, assignment, followup):
    exercise_values = {row['id']: row for row in exercises}
    ex_finding = 'Độ chính xác trung bình trên tập test: ' + '; '.join(
        f'{name} {accuracy_text(exercise_values[key]["test_accuracy_mean"], exercise_values[key]["test_accuracy_std"])}'
        for key, name in [('e1_cnn', 'CNN'), ('e1_mlp', 'MLP'), ('e3_gru_rows', 'GRU đọc hàng')]) + '.'
    groups = {row['experiment']: row for row in assignment['groups']}
    swin = groups['swin_t_full']['test_accuracy']
    resnet = groups['resnet18_full']['test_accuracy']
    finding = (f'Trong lịch 20 epoch, Swin-T full fine-tuning đạt {accuracy_text(swin["mean"], swin["sd"])}, '
               f'còn ResNet18 ở cùng chế độ đạt {accuracy_text(resnet["mean"], resnet["sd"])}.')
    followup_groups = {row['experiment']: row for row in followup['groups']}
    scratch = followup_groups['swin_t_scratch']['test_accuracy']
    full = followup_groups['swin_t_full']['test_accuracy']
    followup_finding = (f'Trong lịch 40 epoch, Swin-T khởi tạo ngẫu nhiên đạt {accuracy_text(scratch["mean"], scratch["sd"])}; '
                        f'bản khởi tạo ImageNet đạt {accuracy_text(full["mean"], full["sd"])}.')
    return [('EX_FINDING', html.escape(ex_finding)), ('A1_FINDING', html.escape(finding)),
            ('A1_FOLLOWUP_FINDING', html.escape(followup_finding)),
            ('A1_FEATURE', f'{100 * swin["mean"]:.2f}'.replace('.', ',') + '<small>%</small>'),
            ('A1_FEATURE_SD', f'Độ lệch chuẩn: {100 * swin["sd"]:.2f} điểm phần trăm'.replace('.', ','))]


def build_site(output: Path) -> Path:
    output = Path(output).resolve()
    protected = [ROOT, ROOT / 'site', ROOT / 'reports']
    if any(output == p or (p != ROOT and p in output.parents) for p in protected):
        raise ValueError('Use a separate build directory, not the source directories.')
    output.mkdir(parents=True, exist_ok=True)
    ex = ROOT / 'reports/exercises'
    a1 = ROOT / 'reports/assignment1'
    a2 = ROOT / 'reports/assignment2'
    summaries = json.loads((ex / 'notebook-evidence/summary.json').read_text())
    assignment = json.loads((a1 / 'notebook-evidence/summary.json').read_text())
    followup = json.loads((a1 / 'swin-followup/summary.json').read_text())
    if not all(group['train_seeds'] == group['test_seeds'] == [42, 43, 44]
               for group in followup['groups']) or len(followup['groups']) != 2:
        raise ValueError('A1 requires all six held-out Swin-T follow-up evaluations.')
    count = sum(row['n_train_runs'] for row in summaries)
    tested = sum(row['n_test_runs'] for row in summaries)
    status = ('Báo cáo chưa có kết quả thực nghiệm.' if not count else
              f'Đã tổng hợp {count} lượt huấn luyện và {tested} lượt đánh giá trên tập test Fashion-MNIST.')
    groups = assignment['groups'] + followup['groups']
    a1_trained = sum(len(group['train_seeds']) for group in groups)
    a1_tested = sum(len(group['test_seeds']) for group in groups)
    a1_status = f'{a1_trained} lượt huấn luyện · {a1_tested} lượt đánh giá trên tập test'
    replacements = [('STATUS', html.escape(status)), ('A1_STATUS', html.escape(a1_status)),
                    ('RESULTS', result_markup(summaries)), ('A1_RESULTS', a1_table_rows(assignment)),
                    ('A1_FOLLOWUP', a1_table_rows(followup))] + measured_copy(summaries, assignment, followup) + a2_markup(a2)
    for name in ['index.html', 'a1.html', 'a2.html']:
        source = (ROOT / 'site' / name).read_text()
        for token, value in replacements:
            source = re.sub(rf'<!--{token}-->.*?<!--/{token}-->', lambda _: value, source, flags=re.DOTALL)
        (output / name).write_text(source)
    for name in ['styles.css', 'app.js', 'favicon.svg']:
        shutil.copy2(ROOT / 'site' / name, output / name)
    (output / 'reports').mkdir(exist_ok=True)
    for directory, name in [(ex, 'exercises.pdf'), (a1, 'assignment1.pdf'), (a2, 'assignment2.pdf')]:
        shutil.copy2(directory / name, output / 'reports' / name)
    a1_media = output / 'a1-media'
    a1_media.mkdir(exist_ok=True)
    for name in ('class_counts.png', 'augmentation_examples.png'):
        shutil.copy2(a1 / 'notebook-evidence/figures' / name, a1_media / name)
    media = output / 'a2-media'
    media.mkdir(exist_ok=True)
    for name, source in [('dataset.png', 'eda_class_distribution.png'), ('examples.png', 'qualitative_comparison.png')]:
        shutil.copy2(a2 / 'notebook-evidence/figures' / source, media / name)
    shutil.copy2(a2 / 'demo.mp4', media / 'demo.mp4')
    (output / 'results.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(a1 / 'notebook-evidence/summary.json', output / 'a1-results.json')
    shutil.copy2(a1 / 'swin-followup/summary.json', output / 'a1-swin-followup.json')
    for name in ('results.csv', 'summary.csv', 'paper_comparison.csv'):
        shutil.copy2(a2 / 'notebook-evidence' / name, output / ('a2-' + name))
    (output / '.nojekyll').touch()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/site')
    print(build_site(parser.parse_args().output))
