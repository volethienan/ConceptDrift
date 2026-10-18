"""Regenerate every DASP-v2 prepared stream from raw data with one command.

Seed roles follow docs/PREREGISTRATION_DASP_V2.md: the two development
datasets get development seeds 2001-2003 and confirmation seeds 3001-3005,
built in one call so all eight are source- and session-disjoint; held-out
datasets get only the confirmation seeds. Run from the directory that holds
``data/`` (raw CSVs), for example the repository root:

    python tools/regenerate_v2_streams.py --output-dir prepared_streams_v2

The output root receives BUILD_PROVENANCE.json (code commit, builder hashes,
raw-data hashes via each manifest, environment, commands, timings) and a
validation report per dataset. Existing output is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_SEEDS = (2001, 2002, 2003)
CONFIRMATION_SEEDS = (3001, 3002, 3003, 3004, 3005)
PLAN = (
    ('nf-bot-iot-v3', 'tools/build_controlled_nf_streams.py', DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS),
    ('nf-unsw-nb15-v3', 'tools/build_controlled_nf_streams.py', DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS),
    ('nf-cse-cic-ids2018-v3', 'tools/build_controlled_nf_streams.py', CONFIRMATION_SEEDS),
    ('nf-ton-iot-v3', 'tools/build_controlled_nf_streams.py', CONFIRMATION_SEEDS),
    ('cic-ids2017', 'tools/build_controlled_cic_streams.py', CONFIRMATION_SEEDS),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(['git', *arguments], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def commands(output: Path, datasets: list[str]) -> list[tuple[str, list[str], tuple[int, ...]]]:
    planned = []
    for dataset, builder, seeds in PLAN:
        if dataset not in datasets:
            continue
        seed_text = ','.join(map(str, seeds))
        command = [sys.executable, str(ROOT / builder), '--stream-seeds', seed_text,
                   '--output-dir', str(output)]
        if builder.endswith('nf_streams.py'):
            command += ['--datasets', dataset]
        planned.append((dataset, command, seeds))
    return planned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output-dir', type=Path, default=Path('prepared_streams_v2'))
    parser.add_argument('--datasets', nargs='+', default=[item[0] for item in PLAN],
                        choices=[item[0] for item in PLAN])
    parser.add_argument('--allow-dirty', action='store_true',
                        help='only for trial builds; a release build must come from a clean commit')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    output = args.output_dir.resolve()
    dirty = git('status', '--porcelain', '--untracked-files=no')
    if dirty and not args.allow_dirty:
        raise SystemExit('commit the builder code first; tracked files are modified:\n' + dirty)
    planned = commands(output, args.datasets)
    for dataset, command, _ in planned:
        print(' '.join(command), flush=True)
        if (output / dataset).exists():
            raise SystemExit(f'{output / dataset} already exists; use a fresh output directory')
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / 'logs').mkdir(exist_ok=True)

    import numpy, pandas, scipy, sklearn  # noqa: E401  recorded, not used

    provenance = {
        'purpose': 'DASP-v2 prepared streams (pre-registration section 2)',
        'code_commit': git('rev-parse', 'HEAD'),
        'tracked_tree_dirty': bool(dirty),
        'builders': {name: sha256(ROOT / name) for name in
                     ('tools/build_controlled_nf_streams.py', 'tools/build_controlled_cic_streams.py',
                      'tools/regenerate_v2_streams.py')},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'numpy': numpy.__version__, 'pandas': pandas.__version__,
                        'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__},
        'working_directory': str(Path.cwd()),
        'seed_roles': {'development': list(DEVELOPMENT_SEEDS), 'confirmation': list(CONFIRMATION_SEEDS)},
        'datasets': {},
    }
    # Builders and the validator import dasp_nids/tools from this checkout.
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(
        [str(ROOT / 'src'), str(ROOT), os.environ.get('PYTHONPATH', '')])}
    failed = False
    for dataset, command, seeds in planned:
        started = time.time()
        with (output / 'logs' / f'{dataset}.log').open('w', encoding='utf-8') as log:
            build = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        validation = None
        if build.returncode == 0:
            report = output / f'validation_{dataset}.json'
            with (output / 'logs' / f'validate_{dataset}.log').open('w', encoding='utf-8') as log:
                validation = subprocess.run(
                    [sys.executable, str(ROOT / 'tools/validate_controlled_streams.py'),
                     '--prepared-dir', str(output), '--datasets', dataset,
                     '--stream-seeds', ','.join(map(str, seeds)), '--output', str(report)],
                    stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        provenance['datasets'][dataset] = {
            'command': command, 'stream_seeds': list(seeds), 'build_returncode': build.returncode,
            'validation_returncode': validation, 'seconds': round(time.time() - started, 1)}
        failed |= build.returncode != 0 or validation != 0
        (output / 'BUILD_PROVENANCE.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        print(f'{dataset}: build={build.returncode} validation={validation}', flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
