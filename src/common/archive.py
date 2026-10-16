"""Explicit access to the immutable pre-refactor training source.

This is for reading published measurements only. Training/resume always hashes
current source and never uses this historical identity.
"""
import hashlib
import json
from pathlib import Path
import subprocess

from common.config import ROOT

RELEASE = '1250e569c6ce055391b5670e2f9a31faedb44b20'
SNAPSHOT = 'docs/training-evidence/source-snapshot.json'


def verified_source(root: Path = ROOT) -> dict[str, str]:
    def blob(path):
        return subprocess.check_output(['git', 'show', f'{RELEASE}:{path}'], cwd=root)
    historical = blob(SNAPSHOT)
    if (root / SNAPSHOT).read_bytes() != historical:
        raise ValueError('Historical source snapshot changed')
    expected = json.loads(historical)['sha256']
    for path, sha in expected.items():
        if hashlib.sha256(blob(path)).hexdigest() != sha:
            raise ValueError(f'Historical source does not match measured snapshot: {path}')
    return expected


def a1_source(root: Path = ROOT) -> dict[str, str]:
    hashes = verified_source(root)
    names = ['__init__.py', 'protocol.py', 'models.py', 'data.py', 'metrics.py',
             'engine.py', 'run_suite.py', 'visuals.py']
    return {key: hashes[key] for key in [*[f'A1/{name}' for name in names], 'common/runtime.py']}
