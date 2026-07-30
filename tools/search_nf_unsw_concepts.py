from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dasp_nids.data import NF_V3_NON_FEATURE_COLUMNS
from tools.build_controlled_nf_streams import (
    DESIGN_SALT,
    _allocate,
    _splitmix64,
)


SUBFOLD_SALT = np.uint64(0x9FB21C651E98DF25)
CANDIDATES: dict[str, dict[str, float]] = {
    "original_B": {
        "generic": 0.45,
        "reconnaissance": 0.45,
        "dos": 0.10,
    },
    "candidate_C": {
        "generic": 0.40,
        "dos": 0.30,
        "backdoor": 0.20,
        "shellcode": 0.10,
    },
    "generic_only": {"generic": 1.0},
    "reconnaissance_only": {"reconnaissance": 1.0},
    "dos_only": {"dos": 1.0},
    "backdoor_only": {"backdoor": 1.0},
    "shellcode_only": {"shellcode": 1.0},
    "generic_backdoor": {"generic": 0.5, "backdoor": 0.5},
    "dos_backdoor": {"dos": 0.5, "backdoor": 0.5},
    "reconnaissance_backdoor": {
        "reconnaissance": 0.5,
        "backdoor": 0.5,
    },
}
A_CONCEPT = {"exploits": 0.5, "fuzzers": 0.5}


def _features(path: Path) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    return [
        name
        for name in columns
        if name not in NF_V3_NON_FEATURE_COLUMNS
        and not name.startswith("__")
    ]


def _collect_design_fold(
    path: Path,
    modulus: int,
    residue: int,
    chunksize: int,
) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    feature_columns = _features(path)
    session_columns = [
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "PROTOCOL",
    ]
    usecols = list(
        dict.fromkeys(
            [
                *feature_columns,
                "Attack",
                *session_columns,
            ]
        )
    )
    parts: dict[str, dict[str, list[np.ndarray]]] = {}
    row_offset = 0
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)
    ):
        row_ids = np.arange(
            row_offset, row_offset + len(chunk), dtype=np.int64
        )
        row_offset += len(chunk)
        session_ids = pd.util.hash_pandas_object(
            chunk[session_columns], index=False
        ).to_numpy(dtype=np.uint64)
        fold = (
            _splitmix64(session_ids ^ DESIGN_SALT)
            % np.uint64(modulus)
        )
        selected = fold == np.uint64(residue)
        if not np.any(selected):
            continue
        families = (
            chunk["Attack"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .to_numpy(dtype=str)
        )
        numeric = (
            chunk.loc[selected, feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32, copy=True)
        )
        numeric[~np.isfinite(numeric)] = np.nan
        selected_positions = np.flatnonzero(selected)
        position_map = np.full(len(chunk), -1, dtype=np.int64)
        position_map[selected_positions] = np.arange(
            len(selected_positions)
        )
        for family in np.unique(families[selected]):
            mask = selected & (families == family)
            local = position_map[np.flatnonzero(mask)]
            target = parts.setdefault(
                family,
                {"x": [], "session_ids": [], "source_ids": []},
            )
            target["x"].append(numeric[local])
            target["session_ids"].append(session_ids[mask])
            target["source_ids"].append(row_ids[mask])
        print(
            f"design scan chunk={chunk_index + 1} rows={row_offset}",
            flush=True,
        )

    pools = {
        family: {
            name: np.concatenate(values)
            for name, values in family_parts.items()
        }
        for family, family_parts in parts.items()
    }
    session_family_count: Counter[int] = Counter()
    for values in pools.values():
        session_family_count.update(
            np.unique(values["session_ids"]).tolist()
        )
    ambiguous = np.asarray(
        [
            session
            for session, count in session_family_count.items()
            if count > 1
        ],
        dtype=np.uint64,
    )
    if len(ambiguous):
        for family, values in pools.items():
            keep = ~np.isin(values["session_ids"], ambiguous)
            pools[family] = {
                name: array[keep] for name, array in values.items()
            }
    return feature_columns, pools


def _split_pool(
    values: dict[str, np.ndarray], split: int
) -> dict[str, np.ndarray]:
    assignment = (
        _splitmix64(values["session_ids"] ^ SUBFOLD_SALT)
        % np.uint64(2)
    )
    mask = assignment == np.uint64(split)
    return {name: array[mask] for name, array in values.items()}


def _sample_mix(
    pools: dict[str, dict[str, np.ndarray]],
    concept: dict[str, float],
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    required = _allocate(count, concept)
    xs: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for family, take in required.items():
        values = pools[family]
        if len(values["x"]) < take:
            raise ValueError(
                f"{family}: design split has {len(values['x'])}, needs {take}"
            )
        selected = rng.choice(len(values["x"]), size=take, replace=False)
        xs.append(values["x"][selected])
        ids.append(values["source_ids"][selected])
    return np.concatenate(xs), np.concatenate(ids)


def _probe_models(seed: int) -> dict[str, object]:
    return {
        "logistic": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def _metrics(
    model: object, x: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    prediction = np.asarray(model.predict(x), dtype=np.int64)
    return {
        "macro_f1": float(
            f1_score(
                y,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "attack_recall": float(
            recall_score(y, prediction, pos_label=1, zero_division=0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session-disjoint NF-UNSW endpoint construct search."
    )
    parser.add_argument(
        "--input",
        default="data/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/nf_unsw_redesign/concept_search",
    )
    parser.add_argument("--design-fold-modulus", type=int, default=5)
    parser.add_argument("--design-fold-residue", type=int, default=0)
    parser.add_argument("--attack-per-split", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--minimum-drop", type=float, default=0.02)
    parser.add_argument("--minimum-regime-auc", type=float, default=0.65)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_columns, raw_pools = _collect_design_fold(
        source,
        args.design_fold_modulus,
        args.design_fold_residue,
        args.chunksize,
    )
    pools = {
        split: {
            family: _split_pool(values, split)
            for family, values in raw_pools.items()
        }
        for split in [0, 1]
    }
    counts = {
        str(split): {
            family: int(len(values["x"]))
            for family, values in sorted(pools[split].items())
        }
        for split in [0, 1]
    }

    attack_count = args.attack_per_split
    benign_count = int(
        round(attack_count * (1.0 - 0.05398313367920508) / 0.05398313367920508)
    )
    rows: list[dict[str, object]] = []
    candidate_errors: dict[str, str] = {}
    for candidate_name, candidate in CANDIDATES.items():
        try:
            for repetition in range(args.repetitions):
                rng = np.random.default_rng(
                    20260730 + repetition * 104729
                )
                a_train_x, a_train_ids = _sample_mix(
                    pools[0], A_CONCEPT, attack_count, rng
                )
                a_test_x, a_test_ids = _sample_mix(
                    pools[1], A_CONCEPT, attack_count, rng
                )
                b_train_x, b_train_ids = _sample_mix(
                    pools[0], candidate, attack_count, rng
                )
                b_test_x, b_test_ids = _sample_mix(
                    pools[1], candidate, attack_count, rng
                )
                benign_train_x, benign_train_ids = _sample_mix(
                    pools[0], {"benign": 1.0}, benign_count, rng
                )
                benign_test_x, benign_test_ids = _sample_mix(
                    pools[1], {"benign": 1.0}, benign_count, rng
                )
                used_ids = np.concatenate(
                    [
                        a_train_ids,
                        a_test_ids,
                        b_train_ids,
                        b_test_ids,
                        benign_train_ids,
                        benign_test_ids,
                    ]
                )
                if len(np.unique(used_ids)) != len(used_ids):
                    raise ValueError(
                        "source overlap inside construct-design repetition"
                    )

                warmup_x = np.concatenate(
                    [benign_train_x, a_train_x]
                )
                warmup_y = np.concatenate(
                    [
                        np.zeros(len(benign_train_x), dtype=np.int64),
                        np.ones(len(a_train_x), dtype=np.int64),
                    ]
                )
                a_eval_x = np.concatenate([benign_test_x, a_test_x])
                a_eval_y = np.concatenate(
                    [
                        np.zeros(len(benign_test_x), dtype=np.int64),
                        np.ones(len(a_test_x), dtype=np.int64),
                    ]
                )
                b_eval_x = np.concatenate([benign_test_x, b_test_x])
                b_eval_y = np.concatenate(
                    [
                        np.zeros(len(benign_test_x), dtype=np.int64),
                        np.ones(len(b_test_x), dtype=np.int64),
                    ]
                )
                for probe_name, estimator in _probe_models(
                    20260730 + repetition
                ).items():
                    model = make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        estimator,
                    )
                    model.fit(warmup_x, warmup_y)
                    a_metrics = _metrics(model, a_eval_x, a_eval_y)
                    b_metrics = _metrics(model, b_eval_x, b_eval_y)

                    regime = make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        _probe_models(
                            20261730 + repetition
                        )[probe_name],
                    )
                    regime_train_x = np.concatenate(
                        [a_train_x, b_train_x]
                    )
                    regime_train_y = np.concatenate(
                        [
                            np.zeros(len(a_train_x), dtype=np.int64),
                            np.ones(len(b_train_x), dtype=np.int64),
                        ]
                    )
                    regime_test_x = np.concatenate(
                        [a_test_x, b_test_x]
                    )
                    regime_test_y = np.concatenate(
                        [
                            np.zeros(len(a_test_x), dtype=np.int64),
                            np.ones(len(b_test_x), dtype=np.int64),
                        ]
                    )
                    regime.fit(regime_train_x, regime_train_y)
                    regime_score = np.asarray(
                        regime.predict_proba(regime_test_x),
                        dtype=float,
                    )[:, 1]
                    rows.append(
                        {
                            "candidate": candidate_name,
                            "composition": json.dumps(
                                candidate, sort_keys=True
                            ),
                            "repetition": repetition,
                            "probe": probe_name,
                            "A_macro_f1": a_metrics["macro_f1"],
                            "B_macro_f1": b_metrics["macro_f1"],
                            "macro_f1_drop": (
                                a_metrics["macro_f1"]
                                - b_metrics["macro_f1"]
                            ),
                            "A_attack_recall": a_metrics[
                                "attack_recall"
                            ],
                            "B_attack_recall": b_metrics[
                                "attack_recall"
                            ],
                            "attack_recall_drop": (
                                a_metrics["attack_recall"]
                                - b_metrics["attack_recall"]
                            ),
                            "attack_regime_auc": float(
                                roc_auc_score(
                                    regime_test_y, regime_score
                                )
                            ),
                        }
                    )
        except ValueError as error:
            candidate_errors[candidate_name] = str(error)

    frame = pd.DataFrame(rows)
    if not len(frame):
        raise ValueError("No candidate produced a valid design result")
    summary = (
        frame.groupby(["candidate", "probe"], as_index=False)
        .agg(
            median_macro_f1_drop=("macro_f1_drop", "median"),
            minimum_macro_f1_drop=("macro_f1_drop", "min"),
            median_attack_recall_drop=("attack_recall_drop", "median"),
            median_attack_regime_auc=("attack_regime_auc", "median"),
        )
    )
    candidate_gate: dict[str, bool] = {}
    candidate_scores: dict[str, float] = {}
    for candidate, group in summary.groupby("candidate"):
        passed = bool(
            (group["median_macro_f1_drop"] >= args.minimum_drop).all()
            and (
                group["median_attack_regime_auc"]
                >= args.minimum_regime_auc
            ).all()
        )
        candidate_gate[str(candidate)] = passed
        candidate_scores[str(candidate)] = float(
            group["median_macro_f1_drop"].min()
        )
    eligible = [
        candidate
        for candidate, passed in candidate_gate.items()
        if passed
    ]
    selected = (
        max(eligible, key=lambda name: candidate_scores[name])
        if eligible
        else None
    )
    report = {
        "source": str(source.resolve()),
        "feature_count": len(feature_columns),
        "design_fold": {
            "modulus": args.design_fold_modulus,
            "residue": args.design_fold_residue,
            "salt": int(DESIGN_SALT),
        },
        "split_counts": counts,
        "attack_per_split": attack_count,
        "benign_per_split": benign_count,
        "repetitions": args.repetitions,
        "thresholds": {
            "minimum_macro_f1_drop": args.minimum_drop,
            "minimum_attack_regime_auc": args.minimum_regime_auc,
        },
        "candidate_errors": candidate_errors,
        "candidate_gate": candidate_gate,
        "candidate_min_probe_drop": candidate_scores,
        "selected_candidate": selected,
        "selected_composition": (
            CANDIDATES[selected] if selected is not None else None
        ),
        "construct_go": selected is not None,
        "caveat": (
            "Performance degradation is a boundary-impact proxy. With a "
            "fixed labeled dataset it does not by itself prove that P(Y|X) "
            "changed."
        ),
    }
    frame.to_csv(output / "candidate_by_repetition.csv", index=False)
    summary.to_csv(output / "candidate_summary.csv", index=False)
    (output / "concept_search.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
