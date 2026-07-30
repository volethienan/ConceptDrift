from __future__ import annotations

import json
from pathlib import Path


LEVELS = {
    "S4-Low": 0.05,
    "S4-Medium": 0.10,
    "S4-High": 0.20,
}


def main() -> None:
    root = Path("prepared_streams_nf_unsw_pilot/nf-unsw-nb15-v3")
    repaired = 0
    for scenario, fraction in LEVELS.items():
        for path in sorted((root / scenario).glob("*.manifest.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["legacy_cli_local_fraction"] = payload.get(
                "local_fraction_of_attacks", 0.05
            )
            payload["local_fraction_of_attacks"] = fraction
            payload.pop("effective_local_fraction_of_attacks", None)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            repaired += 1
    if repaired != 9:
        raise ValueError(f"Expected 9 S4 manifests, repaired {repaired}")
    print(f"repaired {repaired} S4 manifests", flush=True)


if __name__ == "__main__":
    main()
