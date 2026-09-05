# Revision Environment Setup

Captured and validated on September 1, 2026.

```powershell
python -m venv .venv-revision
.\.venv-revision\Scripts\python.exe -m pip install --upgrade `
  "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
.\.venv-revision\Scripts\python.exe -m pip install `
  -r requirements-revision-lock.txt
.\.venv-revision\Scripts\python.exe -m pip install -e . --no-deps
.\.venv-revision\Scripts\python.exe -m pip check
```

Expected isolation invariant:

```text
include-system-site-packages = false
```

River is intentionally excluded from this environment because no P0/P1 revision condition uses the River baselines. Install the optional `river` extra in a separate compatible environment when reproducing legacy River baselines.
