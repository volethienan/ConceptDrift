from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from tools.run_dasp_v16_five_dataset import (
    CANONICAL_SCENARIOS,
    DATASETS,
)


def _flatten(
    x: np.ndarray, y: np.ndarray, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    return (
        x[start:stop].reshape(-1, x.shape[-1]),
        y[start:stop].reshape(-1),
    )


def _models(seed: int) -> dict[str, object]:
    return {
        "logistic": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=128,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + 1,
        ),
    }


def _macro_f1(model: object, x: np.ndarray, y: np.ndarray) -> float:
    return float(
        f1_score(
            y,
            model.predict(x),
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )


def _attack_recall(model: object, x: np.ndarray, y: np.ndarray) -> float:
    return float(
        recall_score(
            y,
            model.predict(x),
            pos_label=1,
            zero_division=0,
        )
    )


def _cap(
    values: np.ndarray, count: int, seed: int
) -> np.ndarray:
    if len(values) <= count:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), size=count, replace=False)]


def _regime_auc(
    model: object,
    left_train: np.ndarray,
    right_train: np.ndarray,
    left_test: np.ndarray,
    right_test: np.ndarray,
    seed: int,
) -> float:
    left_train = _cap(left_train, 4096, seed)
    right_train = _cap(right_train, 4096, seed + 1)
    left_test = _cap(left_test, 4096, seed + 2)
    right_test = _cap(right_test, 4096, seed + 3)
    train_x = np.concatenate([left_train, right_train])
    train_y = np.concatenate(
        [
            np.zeros(len(left_train), dtype=np.int64),
            np.ones(len(right_train), dtype=np.int64),
        ]
    )
    test_x = np.concatenate([left_test, right_test])
    test_y = np.concatenate(
        [
            np.zeros(len(left_test), dtype=np.int64),
            np.ones(len(right_test), dtype=np.int64),
        ]
    )
    model.fit(train_x, train_y)
    return float(
        roc_auc_score(
            test_y,
            np.asarray(model.predict_proba(test_x), dtype=float)[:, 1],
        )
    )


def _primary_slices(
    scenario: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if scenario in {"S2-Gradual", "S3-Incremental"}:
        return (24, 30), (30, 36)
    if scenario == "S5-Blip":
        return (12, 13), (13, 14)
    if scenario == "S0":
        return (18, 24), (24, 30)
    if scenario == "S6-Combined":
        return (12, 14), (14, 16)
    return (12, 18), (18, 24)


def _subtype_mask(
    subtype: np.ndarray,
    y: np.ndarray,
    families: set[str],
    start: int,
    stop: int,
) -> np.ndarray:
    flat_subtype = subtype[start:stop].reshape(-1)
    flat_y = y[start:stop].reshape(-1)
    return (flat_y == 1) & np.isin(flat_subtype, list(families))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", default="prepared_streams_final")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--scenarios", default=",".join(CANONICAL_SCENARIOS)
    )
    parser.add_argument("--seeds", default="2001,2002,2003")
    parser.add_argument(
        "--output",
        default="artifacts/final_five_dataset/construct_validation.json",
    )
    parser.add_argument("--minimum-drop", type=float, default=0.02)
    parser.add_argument("--minimum-regime-auc", type=float, default=0.65)
    parser.add_argument("--minimum-local-recall-drop", type=float, default=0.10)
    parser.add_argument("--minimum-seed-fraction", type=float, default=2 / 3)
    args = parser.parse_args()

    datasets = [
        value.strip() for value in args.datasets.split(",") if value.strip()
    ]
    scenarios = [
        value.strip() for value in args.scenarios.split(",") if value.strip()
    ]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    root = Path(args.prepared_dir)
    rows: list[dict[str, object]] = []
    aprime_rows: list[dict[str, object]] = []

    for dataset in datasets:
        for scenario in scenarios:
            for seed in seeds:
                path = (
                    root / dataset / scenario / f"stream_seed{seed}.npz"
                )
                manifest = json.loads(
                    path.with_suffix(".manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                definitions = manifest["concept_definitions"]
                with np.load(path, allow_pickle=False) as payload:
                    x = np.asarray(payload["x"], dtype=np.float32)
                    y = np.asarray(payload["y"], dtype=np.int64)
                    subtype = (
                        np.asarray(payload["subtypes"])
                        .astype(str)
                        .astype("<U64")
                    )
                a_train_x, a_train_y = _flatten(x, y, 4, 10)
                a_test_x, a_test_y = _flatten(x, y, 10, 12)
                (b_train_start, b_train_stop), (
                    b_test_start,
                    b_test_stop,
                ) = _primary_slices(scenario)
                b_train_x, b_train_y = _flatten(
                    x, y, b_train_start, b_train_stop
                )
                b_test_x, b_test_y = _flatten(
                    x, y, b_test_start, b_test_stop
                )
                for probe_name, a_model in _models(seed).items():
                    b_model = _models(seed + 100_000)[probe_name]
                    a_model.fit(a_train_x, a_train_y)
                    b_model.fit(b_train_x, b_train_y)
                    a_f1 = _macro_f1(a_model, a_test_x, a_test_y)
                    b_f1 = _macro_f1(a_model, b_test_x, b_test_y)
                    attack_auc = _regime_auc(
                        _models(seed + 200_000)[probe_name],
                        a_train_x[a_train_y == 1],
                        b_train_x[b_train_y == 1],
                        a_test_x[a_test_y == 1],
                        b_test_x[b_test_y == 1],
                        seed + 300_000,
                    )
                    benign_auc = _regime_auc(
                        _models(seed + 400_000)[probe_name],
                        a_train_x[a_train_y == 0],
                        b_train_x[b_train_y == 0],
                        a_test_x[a_test_y == 0],
                        b_test_x[b_test_y == 0],
                        seed + 500_000,
                    )
                    row: dict[str, object] = {
                        "dataset": dataset,
                        "scenario": scenario,
                        "seed": seed,
                        "probe": probe_name,
                        "A_macro_f1": a_f1,
                        "B_macro_f1_from_A_model": b_f1,
                        "A_to_B_macro_f1_drop": a_f1 - b_f1,
                        "A_attack_recall": _attack_recall(
                            a_model, a_test_x, a_test_y
                        ),
                        "B_attack_recall_from_A_model": _attack_recall(
                            a_model, b_test_x, b_test_y
                        ),
                        "attack_regime_auc": attack_auc,
                        "benign_regime_auc": benign_auc,
                        "prediction_disagreement_on_B": float(
                            np.mean(
                                a_model.predict(b_test_x)
                                != b_model.predict(b_test_x)
                            )
                        ),
                    }
                    row["A_to_B_attack_recall_drop"] = float(
                        row["A_attack_recall"]
                        - row["B_attack_recall_from_A_model"]
                    )

                    if scenario.startswith("S7-Local"):
                        local_a = set(definitions["local_A"])
                        local_b = set(definitions["local_B"])
                        a_train_mask = _subtype_mask(
                            subtype, y, local_a, 4, 10
                        )
                        a_test_mask = _subtype_mask(
                            subtype, y, local_a, 10, 12
                        )
                        b_train_mask = _subtype_mask(
                            subtype,
                            y,
                            local_b,
                            b_train_start,
                            b_train_stop,
                        )
                        b_test_mask = _subtype_mask(
                            subtype,
                            y,
                            local_b,
                            b_test_start,
                            b_test_stop,
                        )
                        a_train_all, _ = _flatten(x, y, 4, 10)
                        a_test_all, _ = _flatten(x, y, 10, 12)
                        b_train_all, _ = _flatten(
                            x, y, b_train_start, b_train_stop
                        )
                        b_test_all, _ = _flatten(
                            x, y, b_test_start, b_test_stop
                        )
                        local_a_train = a_train_all[a_train_mask]
                        local_a_test = a_test_all[a_test_mask]
                        local_b_train = b_train_all[b_train_mask]
                        local_b_test = b_test_all[b_test_mask]
                        row["A_local_count"] = len(local_a_test)
                        row["B_local_count"] = len(local_b_test)
                        row["A_local_recall"] = float(
                            np.mean(a_model.predict(local_a_test) == 1)
                        )
                        row["B_local_recall"] = float(
                            np.mean(a_model.predict(local_b_test) == 1)
                        )
                        row["local_recall_drop"] = float(
                            row["A_local_recall"] - row["B_local_recall"]
                        )
                        row["local_regime_auc"] = _regime_auc(
                            _models(seed + 600_000)[probe_name],
                            local_a_train,
                            local_b_train,
                            local_a_test,
                            local_b_test,
                            seed + 700_000,
                        )

                    if scenario == "S6-Combined":
                        c_train_x, c_train_y = _flatten(x, y, 24, 26)
                        c_test_x, c_test_y = _flatten(x, y, 26, 28)
                        b_reference_f1 = _macro_f1(
                            b_model, b_test_x, b_test_y
                        )
                        row["B_to_C_macro_f1_drop"] = float(
                            b_reference_f1
                            - _macro_f1(b_model, c_test_x, c_test_y)
                        )
                        row["B_C_attack_regime_auc"] = _regime_auc(
                            _models(seed + 800_000)[probe_name],
                            b_train_x[b_train_y == 1],
                            c_train_x[c_train_y == 1],
                            b_test_x[b_test_y == 1],
                            c_test_x[c_test_y == 1],
                            seed + 900_000,
                        )

                    rows.append(row)

                    if scenario in {"S4-Recurrent", "S6-Combined"}:
                        start = 24 if scenario == "S4-Recurrent" else 34
                        aprime_x, aprime_y = _flatten(x, y, start, 36)
                        regime_model = _models(seed + 1_000_000)[probe_name]
                        _regime_auc(
                            regime_model,
                            a_train_x[a_train_y == 1],
                            b_train_x[b_train_y == 1],
                            a_test_x[a_test_y == 1],
                            b_test_x[b_test_y == 1],
                            seed + 1_100_000,
                        )
                        a_score = np.asarray(
                            regime_model.predict_proba(
                                a_test_x[a_test_y == 1]
                            )
                        )[:, 1].mean()
                        b_score = np.asarray(
                            regime_model.predict_proba(
                                b_test_x[b_test_y == 1]
                            )
                        )[:, 1].mean()
                        ap_score = np.asarray(
                            regime_model.predict_proba(
                                aprime_x[aprime_y == 1]
                            )
                        )[:, 1].mean()
                        aprime_rows.append(
                            {
                                "dataset": dataset,
                                "scenario": scenario,
                                "seed": seed,
                                "probe": probe_name,
                                "A_mean_B_probability": float(a_score),
                                "B_mean_B_probability": float(b_score),
                                "A_prime_mean_B_probability": float(ap_score),
                                "A_prime_closer_to_A": bool(
                                    abs(ap_score - a_score)
                                    < abs(ap_score - b_score)
                                ),
                            }
                        )

    frame = pd.DataFrame(rows)
    aprime = pd.DataFrame(aprime_rows)
    summaries: list[dict[str, object]] = []
    for (dataset, scenario, probe), group in frame.groupby(
        ["dataset", "scenario", "probe"], sort=True
    ):
        if scenario == "S0":
            eligible = True
        elif scenario.startswith("S7-Local"):
            eligible = bool(
                group["local_recall_drop"].median()
                >= args.minimum_local_recall_drop
                and (
                    group["local_recall_drop"]
                    >= args.minimum_local_recall_drop
                ).mean()
                >= args.minimum_seed_fraction
                and group["local_regime_auc"].median()
                >= args.minimum_regime_auc
            )
        else:
            eligible = bool(
                group["A_to_B_macro_f1_drop"].median()
                >= args.minimum_drop
                and (
                    group["A_to_B_macro_f1_drop"]
                    >= args.minimum_drop
                ).mean()
                >= args.minimum_seed_fraction
                and group["attack_regime_auc"].median()
                >= args.minimum_regime_auc
            )
            if scenario == "S6-Combined":
                eligible = bool(
                    eligible
                    and group["B_to_C_macro_f1_drop"].median()
                    >= args.minimum_drop
                    and group["B_C_attack_regime_auc"].median()
                    >= args.minimum_regime_auc
                )
        summaries.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "probe": probe,
                "median_A_to_B_macro_f1_drop": float(
                    group["A_to_B_macro_f1_drop"].median()
                ),
                "median_A_to_B_attack_recall_drop": float(
                    group["A_to_B_attack_recall_drop"].median()
                ),
                "median_attack_regime_auc": float(
                    group["attack_regime_auc"].median()
                ),
                "median_benign_regime_auc": float(
                    group["benign_regime_auc"].median()
                ),
                "median_prediction_disagreement_on_B": float(
                    group["prediction_disagreement_on_B"].median()
                ),
                "median_local_recall_drop": (
                    float(group["local_recall_drop"].median())
                    if "local_recall_drop" in group
                    else None
                ),
                "claim_eligible": eligible,
            }
        )
    summary = pd.DataFrame(summaries)
    scenario_gate = (
        summary.groupby(["dataset", "scenario"])["claim_eligible"]
        .all()
        .to_dict()
    )
    aprime_gate = (
        aprime.groupby(["dataset", "scenario"])["A_prime_closer_to_A"]
        .all()
        .to_dict()
        if len(aprime)
        else {}
    )
    report = {
        "thresholds": {
            "minimum_macro_f1_drop": args.minimum_drop,
            "minimum_attack_regime_auc": args.minimum_regime_auc,
            "minimum_local_recall_drop": args.minimum_local_recall_drop,
            "minimum_seed_fraction": args.minimum_seed_fraction,
        },
        "scenario_gate": {
            f"{dataset}/{scenario}": bool(value)
            for (dataset, scenario), value in scenario_gate.items()
        },
        "A_prime_gate": {
            f"{dataset}/{scenario}": bool(value)
            for (dataset, scenario), value in aprime_gate.items()
        },
        "pilot_construct_go": bool(
            scenario_gate
            and all(scenario_gate.values())
            and all(aprime_gate.values())
        ),
        "summary": summaries,
        "interpretation": (
            "Temporal schedules are valid independently of this gate. A failed "
            "gate is retained as subtype/covariate shift or a negative "
            "construct result and must not be called natural real drift."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_name("construct_by_seed.csv"), index=False)
    aprime.to_csv(output.with_name("aprime_validation.csv"), index=False)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
