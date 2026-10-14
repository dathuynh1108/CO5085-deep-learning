"""Collect completed coursework, verify every file, then optionally delete Vast.

No credentials are stored. --delete is for the user's explicitly authorized cleanup.
Run only after both suites finish; there is no deletion path for incomplete results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def validate_completion(root: Path) -> dict:
    import torch
    from common.engine import source_fingerprint
    from common.runtime import state_fingerprint
    current_source = source_fingerprint()
    counts = {}
    for config_path, runs_path, expected in [('configs/experiments.json', 'runs', 60)]:
        configs = json.loads((root / config_path).read_text())
        count = 0
        for config in configs:
            for seed in [42, 43, 44]:
                run = root / runs_path / config['id'] / f'seed_{seed}'
                if (run / '.train.lock').exists():
                    raise ValueError(f'Active training lock: {run}')
                metrics = json.loads((run / 'metrics.json').read_text())
                evaluation = json.loads((run / 'evaluation_test.json').read_text())
                if metrics.get('status') != 'trained' or metrics['completed_epochs'] != config['epochs']:
                    raise ValueError(f'Incomplete training: {run}')
                if metrics.get('experiment', metrics.get('config')) != config:
                    raise ValueError(f'Unexpected configuration: {run}')
                if metrics['signature'] != evaluation['signature']:
                    raise ValueError(f'Evaluation signature mismatch: {run}')
                if metrics['best_model_sha256'] != evaluation['model_sha256']:
                    raise ValueError(f'Evaluation checkpoint mismatch: {run}')
                if evaluation['split'] != 'test' or evaluation['seed'] != seed:
                    raise ValueError(f'Unexpected evaluation: {run}')
                for filename in ['run.json', 'history.csv', 'best.pt', 'last.pt']:
                    if not (run / filename).is_file():
                        raise ValueError(f'Missing artifact: {run / filename}')
                info = json.loads((run / 'run.json').read_text())
                if info['source_fingerprint'] != current_source:
                    raise ValueError(f'Training source mismatch: {run}')
                checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
                if state_fingerprint(checkpoint['model']) != evaluation['model_sha256']:
                    raise ValueError(f'Actual checkpoint tensor mismatch: {run}')
                if evaluation['n_samples'] != 10000:
                    raise ValueError(f'Unexpected held-out test size: {run}')
                count += 1
        if count != expected:
            raise ValueError(f'Coverage changed: {runs_path}, {count} != {expected}')
        counts[runs_path] = count
    from A1.export_report import read_run
    protocol = json.loads((root / 'configs/a1_experiments.json').read_text())
    a1_count = 0
    for config in protocol['experiments']:
        for seed in [42, 43, 44]:
            run = root / 'A1/runs' / config['id'] / f'seed_{seed}'
            if (run / '.a1.lock').exists():
                raise ValueError(f'Active training/evaluation lock: {run}')
            row, missing = read_run(run, config, seed, protocol)
            if row is None or missing != 'complete' or row['test_accuracy'] is None:
                raise ValueError(f'Incomplete A1 train/test: {run}; {missing}')
            for filename in ['best.pt', 'last.pt', 'metadata.json', 'split.json', 'history.csv']:
                if not (run / filename).is_file():
                    raise ValueError(f'Missing A1 artifact: {run / filename}')
            # The exporter checks the test identity and selection. Verify the
            # complete best.pt file too before trusting the remote backup.
            evaluation = json.loads((run / 'evaluation_test.json').read_text())
            if file_digest(run / 'best.pt') != evaluation['identity']['best_file_sha256']:
                raise ValueError(f'A1 checkpoint file checksum mismatch: {run}')
            a1_count += 1
    if a1_count != 24:
        raise ValueError(f'A1 coverage changed: {a1_count} != 24')
    counts['A1/runs'] = a1_count
    return counts


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_with_retry(command: list[str]) -> None:
    """Resume a transfer after an SSH disconnect; all other failures stop cleanup."""
    for attempt in range(1, 4):
        try:
            subprocess.run(command, check=True, timeout=1800)
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 255 or attempt == 3:
                raise
            print(f'SSH transfer disconnected; retrying {attempt + 1}/3.', flush=True)
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', required=True, type=int)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--identity', required=True, type=Path)
    parser.add_argument('--remote-root', default='/workspace/CO5085-deep-learning')
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
           '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
           '-i', str(args.identity), '-p', str(args.port)]
    target = f'root@{args.host}'
    # Execute the same completion guard remotely before making a snapshot.
    subprocess.run(ssh + [target, f'cd {shlex.quote(args.remote_root)} && /venv/main/bin/python -c '
                           + shlex.quote('from pathlib import Path; from scripts.vast_finalize import validate_completion; '
                                         'print(validate_completion(Path(".")))')], check=True)
    remote_code = ('import hashlib,json,pathlib; root=pathlib.Path(' + repr(args.remote_root) + '); '
                   'files=[p for folder in ["runs","A1/runs","data/processed","logs"] '
                   'for p in (root/folder).rglob("*") if p.is_file()]; '
                   'print(json.dumps({str(p.relative_to(root)):hashlib.file_digest(p.open("rb"),"sha256").hexdigest() '
                   'for p in files},sort_keys=True))')
    hashes = json.loads(subprocess.check_output(ssh + [target, '/venv/main/bin/python -c ' + shlex.quote(remote_code)], text=True))
    for folder in ['runs', 'A1/runs', 'data/processed', 'logs']:
        destination = ROOT / folder
        destination.mkdir(parents=True, exist_ok=True)
        copy_with_retry(['rsync', '-az', '--partial-dir=.rsync-partial', '-e', shlex.join(ssh),
                         f'{target}:{args.remote_root}/{folder}/', str(destination) + '/'])
    mismatches = [name for name, digest in hashes.items()
                  if not (ROOT / name).is_file() or file_digest(ROOT / name) != digest]
    if mismatches:
        raise ValueError(f'Backup checksum verification failed: {mismatches[:10]}')
    coverage = validate_completion(ROOT)
    evidence = ROOT / 'docs/training-evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = {'instance': args.instance, 'coverage': coverage, 'verified_files': len(hashes),
                'sha256': hashes, 'instance_deleted': False}
    path = evidence / 'backup-manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'BACKUP VERIFIED: {coverage}; {len(hashes)} files.', flush=True)
    if args.delete:
        # Match the current rental to the SSH endpoint; never destroy a new unrelated rental.
        snapshot = json.loads(subprocess.check_output(['vastai', 'show', 'instances-v1', '--raw'], text=True))
        candidates = [x for x in snapshot['instances'] if x['id'] == args.instance]
        if len(candidates) != 1:
            raise ValueError('Target instance missing or ambiguous; automatic deletion refused.')
        instance = candidates[0]
        direct_ports = instance.get('ports', {}).get('22/tcp', [])
        direct_match = (instance.get('public_ipaddr') == args.host
                        and any(int(p['HostPort']) == args.port for p in direct_ports))
        proxy_match = instance.get('ssh_host') == args.host and instance.get('ssh_port') == args.port
        if not direct_match and not proxy_match:
            raise ValueError('Target instance endpoint changed; automatic deletion refused.')
        result = subprocess.run(['vastai', 'destroy', 'instance', str(args.instance), '-y', '--raw'],
                                check=True, text=True, capture_output=True)
        remaining = json.loads(subprocess.check_output(['vastai', 'show', 'instances-v1', '--raw'], text=True))
        if any(x['id'] == args.instance for x in remaining.get('instances', [])):
            raise RuntimeError('Vast still lists the instance; deletion not confirmed.')
        manifest.update(instance_deleted=True, deletion_exit_code=result.returncode)
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'INSTANCE {args.instance} DELETED AND VERIFIED.', flush=True)


if __name__ == '__main__':
    main()
