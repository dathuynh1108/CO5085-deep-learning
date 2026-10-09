"""Assemble a static Pages directory using the current report and real run counts."""
from pathlib import Path
import argparse
import html
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def build_site(output: Path) -> Path:
    output = Path(output).resolve()
    if output == ROOT or output in [ROOT / 'site', ROOT / 'reports']:
        raise ValueError('Use a separate build directory, not the source directories.')
    output.mkdir(parents=True, exist_ok=True)
    pdf = ROOT / 'reports/exercises.pdf'
    manifest = json.loads((ROOT / 'reports/generated/manifest.json').read_text(encoding='utf-8'))
    if not pdf.exists():
        raise FileNotFoundError('Build reports/exercises.pdf first.')
    train, test = manifest['train_runs'], manifest['test_runs']
    status = ('Chưa huấn luyện. Chưa có kết quả thực nghiệm.' if train == 0 else
              f'Đã tổng hợp {train} lượt train và {test} lượt test. Xem phạm vi seed trong báo cáo.')
    source = (ROOT / 'site/index.html').read_text(encoding='utf-8')
    source = re.sub(r'<!--STATUS-->.*?<!--/STATUS-->', html.escape(status), source, flags=re.DOTALL)
    (output / 'index.html').write_text(source, encoding='utf-8')
    shutil.copy2(ROOT / 'site/styles.css', output / 'styles.css')
    (output / 'reports').mkdir(exist_ok=True)
    shutil.copy2(pdf, output / 'reports/exercises.pdf')
    shutil.copy2(ROOT / 'reports/generated/summary.json', output / 'results.json')
    (output / '.nojekyll').touch()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'site-preview')
    print(build_site(parser.parse_args().output))


if __name__ == '__main__':
    main()
