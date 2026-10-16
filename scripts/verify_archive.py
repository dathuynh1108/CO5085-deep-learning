"""Verify moved historical source, evidence and optional full local backups."""
import argparse
import hashlib
import json
from pathlib import Path
from common.archive import RELEASE, verified_source
from common.config import ROOT


def relocated(path: str) -> str:
    for old, new in [('A1/reports/', 'reports/assignment1/'),
                     ('reports/', 'reports/exercises/'),
                     ('A1/runs/', 'runs/assignment1/'),
                     ('runs/', 'runs/exercises/'), ('logs/', 'artifacts/logs/')]:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def verify(checkpoints=False):
    sources = verified_source()
    backup = json.loads((ROOT / 'docs/training-evidence/backup-manifest.json').read_text())
    files = backup['sha256'] if 'sha256' in backup else backup['files']
    evidence, full = 0, 0
    for old, expected in files.items():
        # Full checkpoints remain private. Every committed evidence file has a
        # counterpart in the byte-for-byte backup manifest.
        if old.startswith('A1/runs/'):
            light = 'reports/assignment1/generated/evidence/' + old[len('A1/runs/'):]
        elif old.startswith('runs/'):
            light = 'reports/exercises/generated/evidence/' + old[len('runs/'):]
        else: light = None
        if light and (ROOT / light).is_file():
            if digest(ROOT / light) != expected: raise ValueError(f'Altered evidence: {light}')
            evidence += 1
        if checkpoints:
            current = ROOT / relocated(old)
            if digest(current) != expected: raise ValueError(f'Altered backup: {current}')
            full += 1
    result = dict(source_release=RELEASE, source_files=len(sources),
                  verified_evidence_files=evidence, verified_backup_files=full,
                  training_resumed=False)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', action='store_true')
    verify(parser.parse_args().checkpoints)
