"""Package reviewable source, hashes and local validation without data or outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ('src/', 'tools/', 'tests/', 'configs/', 'docs/')
TOP = {'README.md','pyproject.toml','requirements-revision-lock.txt','requirements-original-review-linux.txt'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    files = []
    for prefix in PREFIXES:
        for path in sorted((ROOT/prefix).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'):
                files.append(path)
    files += [ROOT/name for name in sorted(TOP) if (ROOT/name).is_file()]
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    entries = {str(path.relative_to(ROOT)).replace('\\','/'):{'sha256':digest(path),'bytes':path.stat().st_size}
               for path in files}
    manifest = {'branch':branch,'git_commit':commit,'files':entries,
        'data_included':False,'primary_design_changed':False,
        'nvidia_cuda_test_host':'Windows; Linux target not directly executed',
        'main_entrypoint':'python tools/run_original_review.py',
        'guide':'docs/DASP_ORIGINAL_REVIEW.md'}
    evidence = [
        ROOT/'outputs_original_review_validation/release_suite.xml',
        ROOT/'outputs_original_review_gpu_release/smoke_analysis.json',
        ROOT/'outputs_original_review_gate_release/gate_analysis.json',
        ROOT/'outputs_original_review_validation/ENVIRONMENT.json',
    ]
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,str(path.relative_to(ROOT)).replace('\\','/'))
        for path in evidence:
            if path.is_file(): archive.write(path,'validation/'+path.name)
        archive.writestr('RELEASE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(destination) as archive:
        failures = []
        for name,entry in entries.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=entry['sha256']: failures.append(name)
        if failures: raise RuntimeError(f'package hash mismatch: {failures}')
    receipt = {'zip':str(destination.resolve()),'sha256':digest(destination),
        'source_files':len(files),'bytes':destination.stat().st_size,'git_commit':commit,'branch':branch}
    destination.with_suffix('.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/dasp-original-review-server.zip')
    build(parser.parse_args().output)
