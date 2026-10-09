"""Create the NEW coursework repository using the user's authenticated GitHub CLI.

This is an explicit publishing action. It is not called by tests or by training.
The script never overwrites a remote, changes visibility of an existing repo,
uses force push, embeds credentials, or modifies the reference repository.
"""
from pathlib import Path
import argparse
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]
OWNER = 'dathuynh1108'
NAME = 'CO5085-deep-learning'


def create_command(owner: str, visibility: str) -> list[str]:
    if owner != OWNER:
        raise ValueError('This package and its Pages links target dathuynh1108 only.')
    if visibility not in {'public', 'private'}:
        raise ValueError('Choose public or private explicitly.')
    return ['gh', 'repo', 'create', f'{owner}/{NAME}', '--'+visibility,
            '--source', str(ROOT), '--remote', 'origin', '--push']


def run(command, **kwargs):
    return subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', default=OWNER)
    parser.add_argument('--visibility', choices=['public', 'private'], required=True)
    args = parser.parse_args()
    command = create_command(args.owner, args.visibility)
    for tool in ['git', 'gh']:
        if not shutil.which(tool):
            raise SystemExit(f'{tool} is missing. Install it and authenticate gh before publishing.')
    run(['gh', 'auth', 'status'])
    login = run(['gh', 'api', 'user', '--jq', '.login'], capture_output=True, text=True).stdout.strip()
    if login.lower() != args.owner.lower():
        raise SystemExit(f'Authenticated as {login}, expected {args.owner}. No repository was created.')
    exists = subprocess.run(['gh', 'api', f'repos/{args.owner}/{NAME}'], cwd=ROOT,
                            capture_output=True, text=True)
    if exists.returncode == 0:
        raise SystemExit('Target already exists. Inspect it and push manually; this script creates only NEW repositories.')
    if 'HTTP 404' not in exists.stderr and '"status": "404"' not in exists.stdout:
        raise SystemExit('Could not safely verify that the target is absent:\n' + exists.stderr)
    # Do not inherit a parent repository when the archive was extracted inside one.
    if not (ROOT / '.git').exists():
        run(['git', 'init', '-b', 'main'])
    current_root = run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True).stdout.strip()
    if Path(current_root).resolve() != ROOT:
        raise SystemExit('Git root differs from this package. Stop before publishing.')
    remotes = run(['git', 'remote'], capture_output=True, text=True).stdout.split()
    if 'origin' in remotes:
        raise SystemExit('origin is already configured. Inspect it rather than replacing it automatically.')
    for key in ['user.name', 'user.email']:
        value = subprocess.run(['git', 'config', '--get', key], cwd=ROOT, capture_output=True, text=True)
        if value.returncode or not value.stdout.strip():
            raise SystemExit(f'Set your own git {key} before committing. No identity is invented by this script.')
    run(['git', 'add', '--all'])
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
    if changed.returncode == 1:
        run(['git', 'commit', '-m', 'Add AI-assisted E1-E3 implementation and untrained report draft'])
    elif changed.returncode != 0:
        raise SystemExit('Could not inspect staged changes.')
    run(['git', 'rev-parse', '--verify', 'HEAD'], capture_output=True)
    branch = run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip()
    if branch != 'main':
        raise SystemExit('Pages is configured for main. Switch/rename the intended branch before publishing.')
    print(f'Creating {args.visibility} repository {args.owner}/{NAME}', flush=True)
    run(command)
    print('Repository created and pushed. Enable Settings > Pages > GitHub Actions, then run the Pages workflow.')


if __name__ == '__main__':
    main()
