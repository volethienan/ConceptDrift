"""Package the raw run directories of each stage as results_raw/<stage>.tar.gz.

    python tools/package_raw_results.py [--results results] [--out results_raw]

Every completed run keeps its config.yaml, provenance.json, summary.json and
windows.csv (per-window metrics), together with the stage-level plan, analysis
and completion files. Shard stdout logs are left out. Absolute paths of the
machine that ran the experiments are rewritten as paths relative to the
repository, so the archives contain no local user or folder names. Unpack an
archive into results/ to run the table scripts in paper/jtit/ without
re-running the experiments.
"""
from __future__ import annotations

import argparse
import io
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# development pilots that were not adopted are left out; pilot2 holds the clean
# development runs of the memory ablation, so it is kept
STAGES = ('verify', 'heldout', 'heldout_cidds', 'tradeoff', 'verify2', 'verify2_heldout', 'verify2_heldout_cidds',
          'audit', 'audit_heldout', 'revision', 'revision_heldout', 'dedup', 'pilot2', 'memory', 'memory_heldout',
          'acceptance', 'acceptance_heldout')
TEXT = {'.yaml', '.yml', '.json', '.csv', '.txt', '.err', '.log'}
# a Windows or POSIX prefix up to the project folder, in plain or JSON-escaped form
PREFIX = re.compile(r'[A-Za-z]:(?:\\\\|\\|/)(?:[^"\'\s,;]*?(?:\\\\|\\|/))?Q!(?:\\\\|\\|/)(?:\.worktrees(?:\\\\|\\|/)dasp-v2(?:\\\\|\\|/))?')


def clean(data: bytes) -> bytes:
    text = data.decode('utf-8', errors='surrogateescape')
    return PREFIX.sub('', text).encode('utf-8', errors='surrogateescape')


def package(stage_dir: Path, target: Path) -> tuple[int, int]:
    files = 0
    with tarfile.open(target, 'w:gz', compresslevel=9) as tar:
        for path in sorted(stage_dir.rglob('*')):
            rel = path.relative_to(stage_dir.parent)
            if not path.is_file() or 'logs' in rel.parts:
                continue
            data = path.read_bytes()
            if path.suffix.lower() in TEXT:
                data = clean(data)
            info = tarfile.TarInfo(rel.as_posix())
            info.size = len(data)
            info.mtime = int(path.stat().st_mtime)
            tar.addfile(info, io.BytesIO(data))
            files += 1
    return files, target.stat().st_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--out', type=Path, default=ROOT / 'results_raw')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        files, size = package(args.results / stage, args.out / f'{stage}.tar.gz')
        print(f'{stage:24s} {files:6d} files {size / 1e6:7.1f} MB', flush=True)
