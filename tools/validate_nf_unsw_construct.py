from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


SCENARIOS = [
    "S0",
    "S1",
    "S2",
    "S3",
    "S4-Low",
    "S4-Medium",
    "S4-High",
]


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
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def _binary_metrics(
    model: object, x: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    prediction = np.asarray(model.predict(x), dtype=np.int64)
    score = np.asarray(model.predict_proba(x), dtype=float)[:, 1]
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
        "pr_auc": float(average_precision_score(y, score)),
        "mcc": float(matthews_corrcoef(y, prediction)),
    }


def _regime_probe(
    model: object,
    a_train: np.ndarray,
    b_train: np.ndarray,
    a_test: np.ndarray,
    b_test: np.ndarray,
) -> dict[str, float]:
    train_x = np.concatenate([a_train, b_train])
    train_y = np.concatenate(
        [
            np.zeros(len(a_train), dtype=np.int64),
            np.ones(len(b_train), dtype=np.int64),
        ]
    )
    test_x = np.concatenate([a_test, b_test])
    test_y = np.concatenate(
        [
            np.zeros(len(a_test), dtype=np.int64),
            np.ones(len(b_test), dtype=np.int64),
        ]
    )
    model.fit(train_x, train_y)
    score = np.asarray(model.predict_proba(test_x), dtype=float)[:, 1]
    prediction = (score >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(test_y, score)),
        "balanced_accuracy": float(
            0.5
            * (
                np.mean(prediction[test_y == 0] == 0)
                + np.mean(prediction[test_y == 1] == 1)
            )
        ),
        "mean_b_probability_on_A": float(np.mean(score[test_y == 0])),
        "mean_b_probability_on_B": float(np.mean(score[test_y == 1])),
    }


def _disagreement(
    left: object, right: object, x: np.ndarray
) -> float:
    return float(np.mean(left.predict(x) != right.predict(x)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct validation for the NF-UNSW controlled pilot."
    )
    parser.add_argument(
        "--prepared-dir", default="prepared_streams_nf_unsw_pilot"
    )
    parser.add_argument("--seeds", default="3101,3102,3103")
    parser.add_argument(
        "--output",
        default="artifacts/nf_unsw_pilot/construct_validation.json",
    )
    parser.add_argument("--minimum-drop", type=float, default=0.02)
    parser.add_argument("--minimum-regime-auc", type=float, default=0.65)
    parser.add_argument(
        "--minimum-seed-fraction", type=float, default=2 / 3
    )
    args = parser.parse_args()

    root = Path(args.prepared_dir) / "nf-unsw-nb15-v3"
    seeds = [int(value) for value in args.seeds.split(",") if value]
    rows: list[dict[str, object]] = []
    aprime_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for seed in seeds:
            path = root / scenario / f"stream_seed{seed}.npz"
            with np.load(path, allow_pickle=False) as payload:
                x = np.asarray(payload["x"], dtype=np.float32)
                y = np.asarray(payload["y"], dtype=np.int64)
                subtype = np.asarray(payload["subtypes"]).astype(str)
            a_train_x, a_train_y = _flatten(x, y, 4, 10)
            a_test_x, a_test_y = _flatten(x, y, 10, 12)
            if scenario == "S2":
                b_train_x, b_train_y = _flatten(x, y, 24, 30)
                b_test_x, b_test_y = _flatten(x, y, 30, 36)
                b_attack_subtype = subtype[30:36].reshape(-1)
            elif scenario == "S0":
                b_train_x, b_train_y = _flatten(x, y, 18, 24)
                b_test_x, b_test_y = _flatten(x, y, 24, 30)
                b_attack_subtype = subtype[24:30].reshape(-1)
            else:
                b_train_x, b_train_y = _flatten(x, y, 12, 18)
                b_test_x, b_test_y = _flatten(x, y, 18, 24)
                b_attack_subtype = subtype[18:24].reshape(-1)

            a_attack_subtype = subtype[10:12].reshape(-1)
            for model_name, a_model in _models(seed).items():
                b_model = _models(seed + 100_000)[model_name]
                a_model.fit(a_train_x, a_train_y)
                b_model.fit(b_train_x, b_train_y)
                a_metrics = _binary_metrics(
                    a_model, a_test_x, a_test_y
                )
                b_metrics = _binary_metrics(
                    a_model, b_test_x, b_test_y
                )
                b_self_metrics = _binary_metrics(
                    b_model, b_test_x, b_test_y
                )

                attack_regime_model = _models(seed + 200_000)[
                    model_name
                ]
                attack_regime = _regime_probe(
                    attack_regime_model,
                    a_train_x[a_train_y == 1],
                    b_train_x[b_train_y == 1],
                    a_test_x[a_test_y == 1],
                    b_test_x[b_test_y == 1],
                )
                benign_regime_model = _models(seed + 300_000)[
                    model_name
                ]
                benign_regime = _regime_probe(
                    benign_regime_model,
                    a_train_x[a_train_y == 0],
                    b_train_x[b_train_y == 0],
                    a_test_x[a_test_y == 0],
                    b_test_x[b_test_y == 0],
                )
                row = {
                    "scenario": scenario,
                    "seed": seed,
                    "probe": model_name,
                    "A_macro_f1": a_metrics["macro_f1"],
                    "B_macro_f1_from_A_model": b_metrics["macro_f1"],
                    "A_to_B_macro_f1_drop": (
                        a_metrics["macro_f1"] - b_metrics["macro_f1"]
                    ),
                    "A_attack_recall": a_metrics["attack_recall"],
                    "B_attack_recall_from_A_model": b_metrics[
                        "attack_recall"
                    ],
                    "A_to_B_attack_recall_drop": (
                        a_metrics["attack_recall"]
                        - b_metrics["attack_recall"]
                    ),
                    "A_pr_auc": a_metrics["pr_auc"],
                    "B_pr_auc_from_A_model": b_metrics["pr_auc"],
                    "A_mcc": a_metrics["mcc"],
                    "B_mcc_from_A_model": b_metrics["mcc"],
                    "B_self_macro_f1": b_self_metrics["macro_f1"],
                    "attack_regime_auc": attack_regime["auc"],
                    "attack_regime_balanced_accuracy": attack_regime[
                        "balanced_accuracy"
                    ],
                    "benign_regime_auc": benign_regime["auc"],
                    "prediction_disagreement_on_A": _disagreement(
                        a_model, b_model, a_test_x
                    ),
                    "prediction_disagreement_on_B": _disagreement(
                        a_model, b_model, b_test_x
                    ),
                }
                if scenario.startswith("S4"):
                    a_local = (
                        (a_test_y == 1)
                        & (a_attack_subtype == "dos")
                    )
                    b_local = (
                        (b_test_y == 1)
                        & (b_attack_subtype == "reconnaissance")
                    )
                    row.update(
                        {
                            "A_local_attack_count": int(a_local.sum()),
                            "B_local_attack_count": int(b_local.sum()),
                            "A_local_attack_recall": float(
                                np.mean(
                                    a_model.predict(a_test_x[a_local])
                                    == 1
                                )
                            ),
                            "B_local_attack_recall": float(
                                np.mean(
                                    a_model.predict(b_test_x[b_local])
                                    == 1
                                )
                            ),
                        }
                    )
                rows.append(row)

                if scenario == "S3":
                    aprime_x, aprime_y = _flatten(x, y, 24, 36)
                    aprime_metrics = _binary_metrics(
                        a_model, aprime_x, aprime_y
                    )
                    aprime_attack_score = np.asarray(
                        attack_regime_model.predict_proba(
                            aprime_x[aprime_y == 1]
                        ),
                        dtype=float,
                    )[:, 1]
                    aprime_rows.append(
                        {
                            "seed": seed,
                            "probe": model_name,
                            "A_prime_macro_f1_from_A_model": (
                                aprime_metrics["macro_f1"]
                            ),
                            "A_prime_attack_recall_from_A_model": (
                                aprime_metrics["attack_recall"]
                            ),
                            "A_prime_mean_B_probability": float(
                                np.mean(aprime_attack_score)
                            ),
                            "A_mean_B_probability": attack_regime[
                                "mean_b_probability_on_A"
                            ],
                            "B_mean_B_probability": attack_regime[
                                "mean_b_probability_on_B"
                            ],
                            "A_prime_closer_to_A": bool(
                                abs(
                                    np.mean(aprime_attack_score)
                                    - attack_regime[
                                        "mean_b_probability_on_A"
                                    ]
                                )
                                < abs(
                                    np.mean(aprime_attack_score)
                                    - attack_regime[
                                        "mean_b_probability_on_B"
                                    ]
                                )
                            ),
                            "prediction_disagreement_on_A_prime": (
                                _disagreement(
                                    a_model, b_model, aprime_x
                                )
                            ),
                        }
                    )

    frame = pd.DataFrame(rows)
    aprime_frame = pd.DataFrame(aprime_rows)
    summaries: list[dict[str, object]] = []
    for (scenario, probe), group in frame.groupby(
        ["scenario", "probe"], sort=True
    ):
        eligible = scenario == "S0" or bool(
            (
                group["A_to_B_macro_f1_drop"]
                >= args.minimum_drop
            ).mean()
            >= args.minimum_seed_fraction
            and group["A_to_B_macro_f1_drop"].median()
            >= args.minimum_drop
            and group["attack_regime_auc"].median()
            >= args.minimum_regime_auc
        )
        summaries.append(
            {
                "scenario": scenario,
                "probe": probe,
                "median_A_to_B_macro_f1_drop": float(
                    group["A_to_B_macro_f1_drop"].median()
                ),
                "median_A_to_B_attack_recall_drop": float(
                    group["A_to_B_attack_recall_drop"].median()
                ),
                "seed_fraction_drop_at_least_threshold": float(
                    (
                        group["A_to_B_macro_f1_drop"]
                        >= args.minimum_drop
                    ).mean()
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
                "claim_eligible": eligible,
            }
        )
    summary_frame = pd.DataFrame(summaries)
    nonstationary = summary_frame[
        summary_frame["scenario"] != "S0"
    ]
    scenario_gate = (
        nonstationary.groupby("scenario")["claim_eligible"]
        .all()
        .to_dict()
    )
    aprime_gate = bool(
        len(aprime_frame)
        and aprime_frame["A_prime_closer_to_A"].all()
    )
    report = {
        "thresholds": {
            "minimum_macro_f1_drop": args.minimum_drop,
            "minimum_attack_regime_auc": args.minimum_regime_auc,
            "minimum_seed_fraction": args.minimum_seed_fraction,
        },
        "summary": summaries,
        "scenario_gate": scenario_gate,
        "A_prime_gate": aprime_gate,
        "pilot_go": bool(
            all(scenario_gate.values()) and aprime_gate
        ),
        "interpretation": (
            "A transition is claim-eligible only when both independent "
            "probes show a median Macro-F1 drop at or above the locked "
            "threshold, at least the locked fraction of data seeds pass, "
            "and attacks are distinguishable by regime. Regime "
            "distinguishability alone is not treated as real concept drift."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_name("construct_by_seed.csv"), index=False)
    aprime_frame.to_csv(
        output.with_name("aprime_validation.csv"), index=False
    )
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["pilot_go"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
