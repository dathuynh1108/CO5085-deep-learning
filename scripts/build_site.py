"""Build the course site from separate reports and verified measured summaries."""
from pathlib import Path
import argparse
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
    'resnet18_scratch': 'ResNet18 — huấn luyện từ đầu',
    'resnet18_head': 'ResNet18 — chỉ học bộ phân loại',
    'resnet18_partial': 'ResNet18 — tinh chỉnh khối cuối',
    'resnet18_full': 'ResNet18 — tinh chỉnh toàn bộ',
    'swin_t_scratch': 'Swin-T — huấn luyện từ đầu',
    'swin_t_head': 'Swin-T — chỉ học bộ phân loại',
    'swin_t_partial': 'Swin-T — tinh chỉnh khối cuối',
    'swin_t_full': 'Swin-T — tinh chỉnh toàn bộ',
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
    summary = json.loads((evidence / 'summary.json').read_text())
    if not summary['train_runs'] == summary['test_runs'] == 2 * len(summary['seeds']):
        raise ValueError('A2 requires complete training and evaluation evidence.')
    names = {'fixed': 'Trộn cố định', 'guided': 'Trộn theo biên'}
    rows = []
    for variant, name in names.items():
        group = summary['groups'][variant]
        values = [accuracy_text(group[key]['mean'], group[key]['sd']).replace('.', ',')
                  for key in ['miou', 'macro_dice', 'transition_iou']]
        rows.append(f'<tr><th scope="row">{name}</th>' +
                    ''.join(f'<td>{value}</td>' for value in values) + '</tr>')
    status = (f'{summary["train_runs"]} lượt huấn luyện và '
              f'{summary["test_runs"]} lượt đánh giá trên tập test.')
    difference = summary['guided_minus_fixed']
    delta = f'{100 * difference["miou"]["mean"]:+.2f}'.replace('.', ',').replace('-', '−')
    edge = f'{100 * difference["transition_iou"]["mean"]:+.2f}'.replace('.', ',').replace('-', '−')
    finding = (f'Chênh lệch trung bình khi chuyển từ trộn cố định sang trộn theo biên là {delta} điểm phần trăm '
               f'mIoU và {edge} điểm phần trăm IoU dải chuyển tiếp. Chênh lệch được tính theo cặp cùng seed.')
    return [('A2_STATUS', html.escape(status)), ('A2_RESULTS', ''.join(rows)),
            ('A2_FINDING', html.escape(finding))]


def build_site(output: Path) -> Path:
    output = Path(output).resolve()
    protected = [ROOT, ROOT / 'site', ROOT / 'src', ROOT / 'reports']
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
                    ('A1_FOLLOWUP', a1_table_rows(followup))] + a2_markup(a2)
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
    media = output / 'a2-media'
    media.mkdir(exist_ok=True)
    for name, source in [('dataset.png', 'notebook-1.png'), ('examples.png', 'notebook-5.png')]:
        shutil.copy2(a2 / 'notebook-evidence/figures' / source, media / name)
    shutil.copy2(a2 / 'demo.mp4', media / 'demo.mp4')
    (output / 'results.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(a1 / 'notebook-evidence/summary.json', output / 'a1-results.json')
    shutil.copy2(a1 / 'swin-followup/summary.json', output / 'a1-swin-followup.json')
    shutil.copy2(a2 / 'notebook-evidence/summary.json', output / 'a2-results.json')
    (output / '.nojekyll').touch()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/site')
    print(build_site(parser.parse_args().output))
