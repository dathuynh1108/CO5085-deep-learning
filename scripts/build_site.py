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
    status = ('Báo cáo chưa có kết quả thực nghiệm.' if train == 0 else
              f'Đã tổng hợp {train} lượt train và {test} lượt test. Xem phạm vi seed trong báo cáo.')
    source = (ROOT / 'site/index.html').read_text(encoding='utf-8')
    source = re.sub(r'<!--STATUS-->.*?<!--/STATUS-->', html.escape(status), source, flags=re.DOTALL)
    (output / 'index.html').write_text(source, encoding='utf-8')
    shutil.copy2(ROOT / 'site/styles.css', output / 'styles.css')
    shutil.copy2(ROOT / 'site/favicon.svg', output / 'favicon.svg')
    (output / 'reports').mkdir(exist_ok=True)
    shutil.copy2(pdf, output / 'reports/exercises.pdf')
    shutil.copy2(ROOT / 'reports/generated/summary.json', output / 'results.json')
    a1_generated = ROOT / 'A1/reports/generated'
    a1_source = ROOT / 'site/a1.html'
    if a1_source.is_file():
        a1 = a1_source.read_text(encoding='utf-8')
        a1_manifest_path = a1_generated / 'manifest.json'
        a1_manifest = json.loads(a1_manifest_path.read_text(encoding='utf-8')) if a1_manifest_path.is_file() else {}
        a1_train, a1_test = a1_manifest.get('trained_runs', 0), a1_manifest.get('tested_runs', 0)
        a1_status = (f'Đã tổng hợp {a1_train} lượt train và {a1_test} lượt test A1. Xem seed và setting trong báo cáo.'
                     if a1_train else 'Chưa có kết quả thực nghiệm A1.')
        a1 = re.sub(r'<!--A1_STATUS-->.*?<!--/A1_STATUS-->', html.escape(a1_status), a1, flags=re.DOTALL)
        (output / 'a1.html').write_text(a1, encoding='utf-8')
        a1_pdf = ROOT / 'A1/reports/assignment1.pdf'
        if a1_pdf.is_file():
            shutil.copy2(a1_pdf, output / 'reports/assignment1.pdf')
        else:
            a1 = (output / 'a1.html').read_text(encoding='utf-8')
            a1 = a1.replace('href="reports/assignment1.pdf"', 'aria-disabled="true"')
            (output / 'a1.html').write_text(a1, encoding='utf-8')
        if (a1_generated / 'summary.json').is_file():
            shutil.copy2(a1_generated / 'summary.json', output / 'a1-results.json')
        else:
            (output / 'a1-results.json').write_text('[]\n', encoding='utf-8')
    (output / '.nojekyll').touch()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'site-preview')
    print(build_site(parser.parse_args().output))


if __name__ == '__main__':
    main()
