"""Compile the common E1–E3 report with XeLaTeX, without changing results."""
import argparse
from pathlib import Path
import shutil
import subprocess
from common.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root', type=Path, default=ROOT / 'reports')
    args = parser.parse_args()
    engine = shutil.which('xelatex')
    if engine is None:
        raise SystemExit('XeLaTeX not found. Install TeX Live (texlive-xetex) or MiKTeX, then retry.')
    directory = args.report_root.resolve()
    if not (directory / 'generated/status.tex').is_file():
        raise SystemExit('Generate tables first: python -m scripts.export_report')
    command = [engine, '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
               '-jobname=exercises', 'report.tex']
    for _ in range(2):
        subprocess.run(command, cwd=directory, check=True)
    print(directory / 'exercises.pdf')


if __name__ == '__main__':
    main()
