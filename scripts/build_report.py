"""Compile a coursework report with XeLaTeX, without changing results."""
import argparse
from pathlib import Path
import shutil
import subprocess
from common.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root', type=Path, default=ROOT / 'reports')
    parser.add_argument('--jobname', default='exercises', help='A1 uses assignment1 with --report-root A1/reports')
    args = parser.parse_args()
    engine = shutil.which('xelatex') or shutil.which('tectonic')
    if engine is None:
        raise SystemExit('XeLaTeX/Tectonic not found. Install a TeX engine, then retry.')
    directory = args.report_root.resolve()
    if not (directory / 'generated/manifest.json').is_file():
        raise SystemExit('Generate the report tables and manifest first.')
    if not args.jobname or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.jobname):
        parser.error('Job name must be a lowercase filename without path separators.')
    if Path(engine).name == 'tectonic':
        subprocess.run([engine, '--keep-logs', '--keep-intermediates', 'report.tex'], cwd=directory, check=True)
        (directory / 'report.pdf').replace(directory / f'{args.jobname}.pdf')
        print(directory / f'{args.jobname}.pdf')
        return
    command = [engine, '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
               f'-jobname={args.jobname}', 'report.tex']
    for _ in range(2):
        subprocess.run(command, cwd=directory, check=True)
    print(directory / f'{args.jobname}.pdf')


if __name__ == '__main__':
    main()
