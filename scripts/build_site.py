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


def result_markup(exercises, assignment):
    rows = []
    for key in ['e1_softmax', 'e1_mlp', 'e1_cnn', 'e3_gru_rows']:
        row = next(r for r in exercises if r['id'] == key)
        mean, sd = row['test_accuracy_mean'], row['test_accuracy_std']
        value = accuracy_text(mean, sd)
        width = 0 if mean is None else mean*100
        rows.append(f'<div class="result-row"><span>{html.escape("GRU (hàng ảnh)" if key == "e3_gru_rows" else row["label"])}</span><div class="result-track"><i style="--score:{width:.3f}%"></i></div><strong>{value}</strong></div>')
    a1 = []
    for row in assignment['groups']:
        mean, sd = row['test_accuracy']['mean'], row['test_accuracy']['sd']
        value = accuracy_text(mean, sd)
        f1 = row['test_macro_f1']['mean']
        a1.append(f'<tr><th scope="row">{html.escape(A1_NAMES[row["experiment"]])}</th><td>{value}</td><td>{"—" if f1 is None else f"{f1:.4f}"}</td></tr>')
    return ''.join(rows), ''.join(a1)


def build_site(output: Path) -> Path:
    output = Path(output).resolve()
    protected = [ROOT, ROOT / 'site', ROOT / 'src', ROOT / 'reports']
    if any(output == p or (p != ROOT and p in output.parents) for p in protected):
        raise ValueError('Use a separate build directory, not the source directories.')
    output.mkdir(parents=True, exist_ok=True)
    ex = ROOT / 'reports/exercises'
    a1 = ROOT / 'reports/assignment1'
    manifest = json.loads((ex / 'generated/manifest.json').read_text())
    am = json.loads((a1 / 'generated/manifest.json').read_text())
    summaries = json.loads((ex / 'generated/summary.json').read_text())
    assignment = json.loads((a1 / 'generated/summary.json').read_text())
    result_rows, a1_rows = result_markup(summaries, assignment)
    count = manifest['train_runs']
    status = ('Báo cáo chưa có kết quả thực nghiệm.' if not count else
              f'Đã tổng hợp {count} lượt huấn luyện và {manifest["test_runs"]} lượt đánh giá trên tập kiểm tra Fashion-MNIST.')
    a1_status = f'{am["trained_runs"]} lượt huấn luyện · {am["tested_runs"]} lượt đánh giá trên tập kiểm tra'
    for name in ['index.html', 'a1.html']:
        source = (ROOT / 'site' / name).read_text()
        for token, value in [('STATUS', html.escape(status)), ('A1_STATUS', html.escape(a1_status)),
                             ('RESULTS', result_rows), ('A1_RESULTS', a1_rows)]:
            source = re.sub(rf'<!--{token}-->.*?<!--/{token}-->', lambda _: value, source, flags=re.DOTALL)
        (output / name).write_text(source)
    for name in ['styles.css', 'app.js', 'favicon.svg']:
        shutil.copy2(ROOT / 'site' / name, output / name)
    (output / 'reports').mkdir(exist_ok=True)
    for origin, name in [(ex / 'exercises.pdf', 'exercises.pdf'), (a1 / 'assignment1.pdf', 'assignment1.pdf')]:
        shutil.copy2(origin, output / 'reports' / name)
    shutil.copy2(ex / 'generated/summary.json', output / 'results.json')
    shutil.copy2(a1 / 'generated/summary.json', output / 'a1-results.json')
    (output / '.nojekyll').touch()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/site')
    print(build_site(parser.parse_args().output))
