from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dasp_nids.metrics import summarize_windows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute derived summary metrics from saved windows.csv files"
    )
    parser.add_argument("output_root")
    args = parser.parse_args()
    root = Path(args.output_root)
    updated = 0
    for windows_path in sorted(root.glob("*/windows.csv")):
        summary_path = windows_path.parent / "summary.json"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(windows_path)
        warmup_windows = int(frame["warmup"].sum())
        derived = summarize_windows(frame.to_dict(orient="records"), warmup_windows)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(derived)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        updated += 1
    print(f"updated_summaries={updated}")


if __name__ == "__main__":
    main()
