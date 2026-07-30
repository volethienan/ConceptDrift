from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dasp_nids.data import NF_V3_NON_FEATURE_COLUMNS


UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
# Locked after a label-blind quota-feasibility search over session hashes.
# It is not selected from model outcomes.
HASH_SALT = np.uint64(1116)
RANK_SALT = np.uint64(0x94D049BB133111EB)
DESIGN_SALT = np.uint64(0xA24BAED4963EE407)
BOT_IOT_SESSION_BUCKET_MS = np.int64(30 * 60 * 1000)
LOCAL_LEVELS = {
    "S4-Low": 0.05,
    "S4-Medium": 0.10,
    "S4-High": 0.20,
    "S7-Local-Low": 0.05,
    "S7-Local-Medium": 0.10,
    "S7-Local-High": 0.20,
}
TEMPORAL_SCENARIOS = (
    "S0",
    "S1-Sudden",
    "S2-Gradual",
    "S3-Incremental",
    "S4-Recurrent",
    "S5-Blip",
    "S6-Combined",
)
CANONICAL_SCENARIOS = (
    *TEMPORAL_SCENARIOS,
    "S7-Local-Low",
    "S7-Local-Medium",
    "S7-Local-High",
)
STRESS_SCENARIOS = {"S5", "S8-Stress"}
LEGACY_SCENARIOS = {"S1", "S2", "S3", "S4-Low", "S4-Medium", "S4-High"}
SCENARIOS = tuple(
    dict.fromkeys(
        [
            *CANONICAL_SCENARIOS,
            "S8-Stress",
            *sorted(LEGACY_SCENARIOS),
            "S5",
        ]
    )
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    total_rows: int
    attack_rows: int
    family_counts: dict[str, int]
    concept_a: dict[str, float]
    concept_b: dict[str, float]
    concept_c: dict[str, float]
    stable: dict[str, float]
    local_a: dict[str, float]
    local_b: dict[str, float]

    @property
    def attack_rate(self) -> float:
        rate = self.attack_rows / self.total_rows
        if not 0.0 < rate < 1.0:
            raise ValueError(f"{self.name}: binary prevalence must be in (0, 1)")
        return float(rate)


@dataclass
class S5ProbeBundle:
    selection_models: list[object]
    audit_models: list[object]

    def benign_scores(self, values: np.ndarray) -> np.ndarray:
        probabilities = [
            np.asarray(model.predict_proba(values), dtype=float)[:, 0]
            for model in self.selection_models
        ]
        return np.median(np.stack(probabilities), axis=0)


DATASETS: dict[str, DatasetSpec] = {
    "nf-unsw-nb15-v3": DatasetSpec(
        name="NF-UNSW-NB15-v3",
        path="data/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv",
        total_rows=2_365_424,
        attack_rows=127_693,
        family_counts={
            "benign": 2_237_731,
            "exploits": 42_748,
            "fuzzers": 33_816,
            "generic": 19_651,
            "reconnaissance": 17_074,
            "dos": 5_980,
            "backdoor": 4_659,
            "shellcode": 2_381,
            "analysis": 1_226,
            "worms": 158,
        },
        concept_a={"exploits": 0.60, "fuzzers": 0.40},
        concept_b={
            "generic": 0.29,
            "reconnaissance": 0.33,
            "dos": 0.11,
            "backdoor": 0.09,
            "shellcode": 0.04,
            "exploits": 0.08,
            "fuzzers": 0.06,
        },
        concept_c={
            "generic": 0.15,
            "reconnaissance": 0.15,
            "dos": 0.10,
            "backdoor": 0.10,
            "shellcode": 0.05,
            "exploits": 0.25,
            "fuzzers": 0.20,
        },
        stable={"exploits": 0.60, "fuzzers": 0.40},
        local_a={"dos": 1.0},
        local_b={"reconnaissance": 1.0},
    ),
    "nf-cse-cic-ids2018-v3": DatasetSpec(
        name="NF-CSE-CIC-IDS2018-v3",
        path="data/NF-CSE-CIC-IDS2018-v3/NF-CICIDS2018-v3.csv",
        total_rows=20_115_529,
        attack_rows=2_600_903,
        family_counts={
            "benign": 17_514_626,
            "ddos_attack-hoic": 1_032_311,
            "ddos_attacks-loic-http": 288_589,
            "dos_attacks-slowhttptest": 105_550,
            "dos_attacks-hulk": 100_076,
            "dos_attacks-goldeneye": 61_300,
            "dos_attacks-slowloris": 36_040,
            "ftp-bruteforce": 386_720,
            "ssh-bruteforce": 188_474,
            "bot": 207_703,
            "infilteration": 188_152,
        },
        concept_a={
            "ddos_attack-hoic": 1 / 6,
            "ddos_attacks-loic-http": 1 / 6,
            "dos_attacks-slowhttptest": 1 / 6,
            "dos_attacks-hulk": 1 / 6,
            "dos_attacks-goldeneye": 1 / 6,
            "dos_attacks-slowloris": 1 / 6,
        },
        concept_b={
            "ftp-bruteforce": 0.25,
            "ssh-bruteforce": 0.25,
            "bot": 0.25,
            "infilteration": 0.25,
        },
        concept_c={"ddos_attack-hoic": 0.5, "bot": 0.5},
        stable={"ddos_attack-hoic": 1.0},
        local_a={"dos_attacks-hulk": 1.0},
        local_b={"ftp-bruteforce": 1.0},
    ),
    "nf-ton-iot-v3": DatasetSpec(
        name="NF-ToN-IoT-v3",
        path="data/NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv",
        total_rows=27_520_260,
        attack_rows=10_728_046,
        family_counts={
            "benign": 16_792_214,
            "ddos": 4_141_256,
            "dos": 203_456,
            "xss": 2_834_435,
            "password": 1_594_777,
            "scanning": 1_358_977,
            "injection": 381_777,
            "backdoor": 203_384,
            "mitm": 6_013,
            "ransomware": 3_971,
        },
        # Twenty-nine percent DoS is the largest rounded mixture that remains
        # feasible for eight source/session-disjoint streams after removing
        # directional 5-tuples carrying multiple attack-family labels.
        concept_a={"ddos": 0.71, "dos": 0.29},
        concept_b={
            "xss": 0.40,
            "password": 0.20,
            "scanning": 0.20,
            "injection": 0.20,
        },
        concept_c={"xss": 0.50, "backdoor": 0.50},
        stable={"ddos": 1.0},
        local_a={"dos": 1.0},
        local_b={"scanning": 1.0},
    ),
    "nf-bot-iot-v3": DatasetSpec(
        name="NF-BoT-IoT-v3",
        path="data/NF-BoT-IoT-v3/NF-BoT-IoT-v3.csv",
        total_rows=16_933_808,
        attack_rows=16_881_819,
        family_counts={
            "benign": 51_989,
            "dos": 8_034_190,
            "ddos": 7_150_882,
            "reconnaissance": 1_695_132,
        },
        concept_a={"ddos": 1.0},
        concept_b={"dos": 1.0},
        concept_c={"reconnaissance": 1.0},
        stable={"dos": 1.0},
        local_a={"reconnaissance": 1.0},
        local_b={"ddos": 1.0},
    ),
}


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
    return values ^ (values >> np.uint64(31))


def _normalise_family(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.strip().str.casefold().to_numpy(dtype=str)


def _allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    names = list(weights)
    probabilities = np.asarray([weights[name] for name in names], dtype=float)
    probabilities /= probabilities.sum()
    expected = probabilities * int(total)
    counts = np.floor(expected).astype(int)
    remainder = int(total) - int(counts.sum())
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:remainder]] += 1
    return {name: int(count) for name, count in zip(names, counts)}


def _window_recipe(
    spec: DatasetSpec,
    scenario: str,
    window: int,
    attack_count: int,
    local_fraction: float,
) -> tuple[dict[str, int], str]:
    def interpolate(
        left: dict[str, float],
        right: dict[str, float],
        alpha: float,
    ) -> dict[str, int]:
        right_count = int(round(float(alpha) * attack_count))
        counts = Counter(_allocate(attack_count - right_count, left))
        counts.update(_allocate(right_count, right))
        return {
            family: int(count)
            for family, count in counts.items()
            if count > 0
        }

    if scenario == "S0":
        return _allocate(attack_count, spec.concept_a), "A"
    if scenario in {"S1", "S1-Sudden"}:
        if window < 12:
            return _allocate(attack_count, spec.concept_a), "A"
        return _allocate(attack_count, spec.concept_b), "B"
    if scenario == "S2-Gradual":
        if window < 12:
            return _allocate(attack_count, spec.concept_a), "A"
        if window >= 24:
            return _allocate(attack_count, spec.concept_b), "B"
        # Gradual drift interleaves pure old/new concepts.  The fraction of
        # B windows rises across four three-window transition blocks.
        b_windows = {14, 16, 18, 19, 21, 22, 23}
        if window in b_windows:
            return _allocate(attack_count, spec.concept_b), "gradual_B"
        return _allocate(attack_count, spec.concept_a), "gradual_A"
    if scenario in {"S2", "S3-Incremental"}:
        if window < 12:
            return _allocate(attack_count, spec.concept_a), "A"
        if window >= 24:
            return _allocate(attack_count, spec.concept_b), "B"
        alpha = (window - 11) / 12
        return (
            interpolate(spec.concept_a, spec.concept_b, alpha),
            f"incremental_{window - 11:02d}",
        )
    if scenario in {"S3", "S4-Recurrent"}:
        if window < 12:
            return _allocate(attack_count, spec.concept_a), "A1"
        if window < 24:
            return _allocate(attack_count, spec.concept_b), "B"
        return _allocate(attack_count, spec.concept_a), "A2"
    if scenario == "S5-Blip":
        if window < 12:
            return _allocate(attack_count, spec.concept_a), "A1"
        if window < 14:
            return _allocate(attack_count, spec.concept_b), "B_blip"
        return _allocate(attack_count, spec.concept_a), "A2"
    if scenario == "S6-Combined":
        if window < 12:
            return _allocate(attack_count, spec.concept_a), "A"
        if window < 16:
            return _allocate(attack_count, spec.concept_b), "B_sudden"
        if window < 24:
            alpha = (window - 15) / 8
            return (
                interpolate(spec.concept_b, spec.concept_c, alpha),
                f"B_to_C_incremental_{window - 15:02d}",
            )
        if window < 28:
            return _allocate(attack_count, spec.concept_c), "C"
        alpha = (window - 27) / 8
        return (
            interpolate(spec.concept_c, spec.concept_a, alpha),
            f"C_to_A_recurrence_{window - 27:02d}",
        )
    if scenario == "S4" or scenario in LOCAL_LEVELS:
        effective_fraction = LOCAL_LEVELS.get(scenario, local_fraction)
        local_count = int(round(attack_count * effective_fraction))
        stable_count = attack_count - local_count
        counts = Counter(_allocate(stable_count, spec.stable))
        counts.update(
            _allocate(
                local_count,
                spec.local_a if window < 12 else spec.local_b,
            )
        )
        return dict(counts), "local_A" if window < 12 else "local_B"
    if scenario in STRESS_SCENARIOS:
        if window < 12:
            return _allocate(attack_count, spec.concept_a), "A"
        return _allocate(attack_count, spec.concept_b), "B_evasive"
    raise ValueError(f"Unknown scenario: {scenario}")


def _requirements(
    spec: DatasetSpec,
    scenarios: tuple[str, ...],
    window_size: int,
    n_windows: int,
    local_fraction: float,
) -> tuple[int, int, dict[str, int]]:
    if n_windows != 36:
        raise ValueError("Controlled protocol requires exactly 36 windows")
    attack_count = int(round(window_size * spec.attack_rate))
    benign_count = window_size - attack_count
    required: dict[str, int] = defaultdict(int)
    for scenario in scenarios:
        scenario_totals: Counter[str] = Counter()
        for window in range(n_windows):
            recipe, _ = _window_recipe(
                spec, scenario, window, attack_count, local_fraction
            )
            scenario_totals.update(recipe)
        for family, count in scenario_totals.items():
            required[family] = max(required[family], int(count))
    required["benign"] = benign_count * n_windows
    return attack_count, benign_count, dict(required)


def _scenario_attack_requirements(
    spec: DatasetSpec,
    scenario: str,
    window_size: int,
    n_windows: int,
    local_fraction: float,
    start_window: int = 0,
) -> dict[str, int]:
    attack_count = int(round(window_size * spec.attack_rate))
    totals: Counter[str] = Counter()
    for window in range(start_window, n_windows):
        recipe, _ = _window_recipe(
            spec, scenario, window, attack_count, local_fraction
        )
        totals.update(recipe)
    return dict(totals)


def _feature_columns(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    columns = [
        str(column)
        for column in header
        if column not in NF_V3_NON_FEATURE_COLUMNS
        and not str(column).startswith("__")
    ]
    if not columns:
        raise ValueError(f"No feature columns found in {path}")
    return columns


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _probe_models(seed: int) -> list[object]:
    return [
        make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=128,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed + 1,
            ),
        ),
        make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=128,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed + 2,
            ),
        ),
    ]


def _design_split(
    spec: DatasetSpec,
    design_pool: dict[str, dict[str, np.ndarray]],
    per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_required = _allocate(per_class, spec.concept_a)
    audit_required = _allocate(per_class, spec.concept_a)
    benign = design_pool["benign"]["x"]
    if len(benign) < 2 * per_class:
        raise ValueError(
            f"{spec.name}: S5 design benign pool has {len(benign)}, "
            f"needs {2 * per_class}"
        )
    train_parts = [benign[:per_class]]
    audit_parts = [benign[per_class : 2 * per_class]]
    train_labels = [np.zeros(per_class, dtype=np.int64)]
    audit_labels = [np.zeros(per_class, dtype=np.int64)]
    train_attack_total = 0
    audit_attack_total = 0
    for family in train_required:
        train_count = train_required[family]
        audit_count = audit_required[family]
        total = train_count + audit_count
        values = design_pool[family]["x"]
        if len(values) < total:
            raise ValueError(
                f"{spec.name}: S5 design pool {family} has {len(values)}, "
                f"needs {total}"
            )
        train_parts.append(values[:train_count])
        audit_parts.append(values[train_count : train_count + audit_count])
        train_labels.append(np.ones(train_count, dtype=np.int64))
        audit_labels.append(np.ones(audit_count, dtype=np.int64))
        train_attack_total += train_count
        audit_attack_total += audit_count
    if train_attack_total != per_class or audit_attack_total != per_class:
        raise AssertionError("S5 design attack allocation is not balanced")
    return (
        np.concatenate(train_parts),
        np.concatenate(train_labels),
        np.concatenate(audit_parts),
        np.concatenate(audit_labels),
    )


def _fit_s5_probes(
    spec: DatasetSpec,
    design_pool: dict[str, dict[str, np.ndarray]],
    per_class: int,
    seed: int,
) -> S5ProbeBundle:
    train_x, train_y, audit_x, audit_y = _design_split(
        spec, design_pool, per_class
    )
    selection_models = _probe_models(seed)
    audit_models = _probe_models(seed + 10_000)
    for model in selection_models:
        model.fit(train_x, train_y)
    for model in audit_models:
        model.fit(audit_x, audit_y)
    return S5ProbeBundle(selection_models, audit_models)


def _select_top_by_time_quantile(
    pool: dict[str, np.ndarray],
    scores: np.ndarray,
    count: int,
    n_quantiles: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if count > len(scores):
        raise ValueError(
            f"S5 candidate pool has {len(scores)} rows, needs {count}"
        )
    if len(scores) != len(pool["x"]):
        raise ValueError("S5 candidate scores do not align with the pool")
    stable_time_order = np.lexsort((pool["source_ids"], pool["timestamps"]))
    quantile_groups = np.array_split(stable_time_order, n_quantiles)
    allocations = _allocate(
        count, {str(index): 1.0 for index in range(n_quantiles)}
    )
    chosen_parts: list[np.ndarray] = []
    for index, group in enumerate(quantile_groups):
        take = allocations[str(index)]
        if take > len(group):
            raise ValueError(
                f"S5 time quantile {index} has {len(group)} rows, needs {take}"
            )
        score_order = np.lexsort(
            (pool["source_ids"][group], -scores[group])
        )
        chosen_parts.append(group[score_order[:take]])
    chosen = np.concatenate(chosen_parts)
    chosen = chosen[rng.permutation(len(chosen))]
    selected = {name: values[chosen] for name, values in pool.items()}
    selected["evasive_score"] = scores[chosen].astype(np.float32)
    return selected


def _select_matched_random_by_time_quantile(
    pool: dict[str, np.ndarray],
    count: int,
    n_quantiles: int,
    excluded_source_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw a time-matched random-B control disjoint from selected S5 rows."""
    stable_time_order = np.lexsort((pool["source_ids"], pool["timestamps"]))
    quantile_groups = np.array_split(stable_time_order, n_quantiles)
    allocations = _allocate(
        count, {str(index): 1.0 for index in range(n_quantiles)}
    )
    excluded = set(np.asarray(excluded_source_ids, dtype=np.int64).tolist())
    chosen_parts: list[np.ndarray] = []
    for index, group in enumerate(quantile_groups):
        eligible = np.asarray(
            [
                row
                for row in group
                if int(pool["source_ids"][row]) not in excluded
            ],
            dtype=np.int64,
        )
        take = allocations[str(index)]
        if take > len(eligible):
            raise ValueError(
                f"S5 matched-control time quantile {index} has "
                f"{len(eligible)} eligible rows, needs {take}"
            )
        chosen_parts.append(
            rng.choice(eligible, size=take, replace=False)
        )
    chosen = np.concatenate(chosen_parts)
    chosen = chosen[rng.permutation(len(chosen))]
    return {name: values[chosen] for name, values in pool.items()}


def _s5_audit(
    raw_x: np.ndarray,
    matched_control_raw_x: np.ndarray,
    labels: np.ndarray,
    audit_models: list[object],
    min_scope: float,
    min_recall_drop: float,
    min_macro_f1_drop: float,
    min_scope_uplift: float,
    min_recall_drop_uplift: float,
    min_macro_f1_drop_uplift: float,
) -> dict[str, object]:
    pre_x = raw_x[7:12].reshape(-1, raw_x.shape[-1])
    pre_y = labels[7:12].reshape(-1)
    post_x = raw_x[12:17].reshape(-1, raw_x.shape[-1])
    control_post_x = matched_control_raw_x[12:17].reshape(
        -1, matched_control_raw_x.shape[-1]
    )
    post_y = labels[12:17].reshape(-1)
    rows: list[dict[str, object]] = []
    post_attack_predictions: list[np.ndarray] = []
    control_attack_predictions: list[np.ndarray] = []
    for model in audit_models:
        pre_prediction = np.asarray(model.predict(pre_x), dtype=np.int64)
        post_prediction = np.asarray(model.predict(post_x), dtype=np.int64)
        control_prediction = np.asarray(
            model.predict(control_post_x), dtype=np.int64
        )
        pre_f1 = float(
            f1_score(
                pre_y,
                pre_prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        post_f1 = float(
            f1_score(
                post_y,
                post_prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        pre_recall = float(
            recall_score(
                pre_y,
                pre_prediction,
                labels=[0, 1],
                average=None,
                zero_division=0,
            )[1]
        )
        post_recall = float(
            recall_score(
                post_y,
                post_prediction,
                labels=[0, 1],
                average=None,
                zero_division=0,
            )[1]
        )
        control_f1 = float(
            f1_score(
                post_y,
                control_prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        control_recall = float(
            recall_score(
                post_y,
                control_prediction,
                labels=[0, 1],
                average=None,
                zero_division=0,
            )[1]
        )
        rows.append(
            {
                "audit_model": type(model[-1]).__name__
                if hasattr(model, "__getitem__")
                else type(model).__name__,
                "pre_macro_f1": pre_f1,
                "post_macro_f1": post_f1,
                "macro_f1_drop": pre_f1 - post_f1,
                "matched_random_post_macro_f1": control_f1,
                "macro_f1_drop_uplift_vs_matched_random": (
                    control_f1 - post_f1
                ),
                "pre_attack_recall": pre_recall,
                "post_attack_recall": post_recall,
                "attack_recall_drop": pre_recall - post_recall,
                "matched_random_post_attack_recall": control_recall,
                "attack_recall_drop_uplift_vs_matched_random": (
                    control_recall - post_recall
                ),
            }
        )
        post_attack_predictions.append(post_prediction[post_y == 1])
        control_attack_predictions.append(
            control_prediction[post_y == 1]
        )
    median_macro_drop = float(
        np.median([row["macro_f1_drop"] for row in rows])
    )
    median_recall_drop = float(
        np.median([row["attack_recall_drop"] for row in rows])
    )
    attack_votes = np.stack(post_attack_predictions)
    control_attack_votes = np.stack(control_attack_predictions)
    consensus_evasion_rate = float(
        np.mean(np.mean(attack_votes == 0, axis=0) >= 0.5)
    )
    control_consensus_evasion_rate = float(
        np.mean(np.mean(control_attack_votes == 0, axis=0) >= 0.5)
    )
    scope_uplift = consensus_evasion_rate - control_consensus_evasion_rate
    median_recall_drop_uplift = float(
        np.median(
            [
                row["attack_recall_drop_uplift_vs_matched_random"]
                for row in rows
            ]
        )
    )
    median_macro_drop_uplift = float(
        np.median(
            [
                row["macro_f1_drop_uplift_vs_matched_random"]
                for row in rows
            ]
        )
    )
    conditions = {
        "consensus_evasion_rate": consensus_evasion_rate >= min_scope,
        "attack_recall_drop": median_recall_drop >= min_recall_drop,
        "macro_f1_drop": median_macro_drop >= min_macro_f1_drop,
        "evasion_uplift_vs_matched_random": scope_uplift >= min_scope_uplift,
        "attack_recall_drop_uplift_vs_matched_random": (
            median_recall_drop_uplift >= min_recall_drop_uplift
        ),
        "macro_f1_drop_uplift_vs_matched_random": (
            median_macro_drop_uplift >= min_macro_f1_drop_uplift
        ),
    }
    return {
        "probe_rows": rows,
        "consensus_evasion_rate": consensus_evasion_rate,
        "matched_random_consensus_evasion_rate": (
            control_consensus_evasion_rate
        ),
        "evasion_uplift_vs_matched_random": scope_uplift,
        "median_attack_recall_drop": median_recall_drop,
        "median_macro_f1_drop": median_macro_drop,
        "median_attack_recall_drop_uplift_vs_matched_random": (
            median_recall_drop_uplift
        ),
        "median_macro_f1_drop_uplift_vs_matched_random": (
            median_macro_drop_uplift
        ),
        "thresholds": {
            "min_consensus_evasion_rate": min_scope,
            "min_attack_recall_drop": min_recall_drop,
            "min_macro_f1_drop": min_macro_f1_drop,
            "min_evasion_uplift_vs_matched_random": min_scope_uplift,
            "min_attack_recall_drop_uplift_vs_matched_random": (
                min_recall_drop_uplift
            ),
            "min_macro_f1_drop_uplift_vs_matched_random": (
                min_macro_f1_drop_uplift
            ),
        },
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
    }


def _collect_pools(
    spec: DatasetSpec,
    stream_seeds: list[int],
    required: dict[str, int],
    s5_candidate_required: dict[str, int],
    s5_design_required: dict[str, int],
    s5_candidate_multiplier: float,
    chunksize: int,
    excluded_design_modulus: int,
    excluded_design_residue: int,
) -> tuple[
    list[str],
    dict[tuple[str, int], dict[str, np.ndarray]],
    dict[tuple[str, int], dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    Counter[str],
]:
    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(path)
    feature_columns = _feature_columns(path)
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
                "FLOW_START_MILLISECONDS",
                *session_columns,
            ]
        )
    )
    n_slots = len(stream_seeds)
    stream_take_by_family: dict[str, int] = {}
    design_take_per_slot: dict[str, int] = {}
    for family in required:
        # Keep a candidate margin so each directional 5-tuple can be
        # assigned to exactly one of warm-up/A/B/final while multiple flows
        # from that session remain usable inside its assigned segment.
        margin = (
            1.16
            if spec.name == "NF-UNSW-NB15-v3" and family == "generic"
            # This is only a temporary candidate-pool reserve. The emitted
            # stream still draws the exact benign quota. A 100% reserve is
            # needed to allocate large whole 5-tuples across four segments
            # without session leakage.
            else (2.00 if family == "benign" else 1.02)
        )
        stream_take = int(np.ceil(margin * required[family]))
        if family in s5_candidate_required:
            target = int(
                np.ceil(
                    s5_candidate_required[family] * s5_candidate_multiplier
                )
            )
            expected_per_slot = spec.family_counts[family] / n_slots
            safe_cap = int(np.floor(0.85 * expected_per_slot))
            if safe_cap >= required[family]:
                target = min(target, safe_cap)
            stream_take = max(required[family], target)
        stream_take_by_family[family] = stream_take
        design_take_per_slot[family] = int(
            np.ceil(s5_design_required.get(family, 0) / n_slots)
        )
    collected: dict[tuple[str, int], dict[str, list[np.ndarray]]] = {}
    for family in required:
        for slot in range(n_slots):
            collected[(family, slot)] = {
                "x": [],
                "source_ids": [],
                "session_ids": [],
                "timestamps": [],
                "rank": [],
            }

    needed_families = set(required)
    observed: Counter[str] = Counter()
    row_offset = 0
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)
    ):
        row_ids = np.arange(row_offset, row_offset + len(chunk), dtype=np.uint64)
        row_offset += len(chunk)
        families = _normalise_family(chunk["Attack"])
        family_counts = Counter(families.tolist())
        for family in needed_families:
            observed[family] += family_counts.get(family, 0)

        session_frame = chunk[session_columns].copy()
        if spec.name == "NF-BoT-IoT-v3":
            session_times = pd.to_numeric(
                chunk["FLOW_START_MILLISECONDS"], errors="coerce"
            ).to_numpy()
            if np.any(~np.isfinite(session_times)):
                raise ValueError(f"Invalid timestamps in {path}")
            session_frame["__TIME_BUCKET_30MIN"] = np.floor_divide(
                session_times.astype(np.int64),
                BOT_IOT_SESSION_BUCKET_MS,
            )
        session_hashes = pd.util.hash_pandas_object(
            session_frame, index=False
        ).to_numpy(dtype=np.uint64)
        if excluded_design_modulus > 1:
            design_folds = (
                _splitmix64(session_hashes ^ DESIGN_SALT)
                % np.uint64(excluded_design_modulus)
            )
            stream_eligible = (
                design_folds
                != np.uint64(excluded_design_residue)
            )
        else:
            stream_eligible = np.ones(len(chunk), dtype=bool)
        assignment_hash = _splitmix64(session_hashes ^ HASH_SALT)
        slots = assignment_hash % np.uint64(n_slots)
        ranks = _splitmix64(row_ids ^ RANK_SALT)
        selected_masks: dict[tuple[str, int], np.ndarray] = {}
        union = np.zeros(len(chunk), dtype=bool)
        for family in needed_families:
            source_count = spec.family_counts[family]
            available_fraction = (
                1.0
                if excluded_design_modulus <= 1
                else (excluded_design_modulus - 1)
                / excluded_design_modulus
            )
            expected_per_slot = (
                source_count * available_fraction / n_slots
            )
            collect_take = (
                stream_take_by_family[family]
                + design_take_per_slot[family]
            )
            probability = min(
                1.0,
                (collect_take / max(1.0, expected_per_slot)) * 1.25
                + 0.01,
            )
            if family != "benign":
                probability = 1.0
            rank_limit = np.uint64(
                min((2**64) - 1, int(probability * ((2**64) - 1)))
            )
            family_mask = (families == family) & stream_eligible
            if not np.any(family_mask):
                continue
            for slot in range(n_slots):
                mask = (
                    family_mask
                    & (slots == np.uint64(slot))
                    & (ranks <= rank_limit)
                )
                if np.any(mask):
                    selected_masks[(family, slot)] = mask
                    union |= mask
        if np.any(union):
            numeric = (
                chunk.loc[union, feature_columns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32, copy=True)
            )
            union_positions = np.flatnonzero(union)
            position_map = np.full(len(chunk), -1, dtype=np.int64)
            position_map[union_positions] = np.arange(len(union_positions))
            timestamps = pd.to_numeric(
                chunk["FLOW_START_MILLISECONDS"], errors="coerce"
            ).to_numpy()
            if np.any(~np.isfinite(timestamps[union])):
                raise ValueError(f"Invalid timestamps in {path}")
            for key, mask in selected_masks.items():
                local = position_map[np.flatnonzero(mask)]
                collected[key]["x"].append(numeric[local])
                collected[key]["source_ids"].append(
                    row_ids[mask].astype(np.int64)
                )
                collected[key]["session_ids"].append(
                    session_hashes[mask]
                )
                collected[key]["timestamps"].append(
                    timestamps[mask].astype(np.int64)
                )
                collected[key]["rank"].append(ranks[mask])
        print(
            f"scan dataset={spec.name} chunk={chunk_index + 1} rows={row_offset}",
            flush=True,
        )

    if row_offset != spec.total_rows:
        raise ValueError(
            f"{spec.name}: expected {spec.total_rows} rows, observed {row_offset}"
        )
    for family in needed_families:
        expected = spec.family_counts[family]
        if observed[family] != expected:
            raise ValueError(
                f"{spec.name}/{family}: expected {expected}, observed {observed[family]}"
            )

    pools: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    candidate_pools: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}
    design_parts: dict[str, dict[str, list[np.ndarray]]] = {
        family: {"x": [], "source_ids": [], "timestamps": []}
        for family in s5_design_required
    }
    merged_by_key: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}
    for key, values in collected.items():
        family, _ = key
        if not values["x"]:
            available = 0
            raise ValueError(
                f"{spec.name}/{family}: collected {available}, need {required[family]}"
            )
        merged_by_key[key] = {
            name: np.concatenate(parts) for name, parts in values.items()
        }

    # Remove directional 5-tuples that carry more than one family, then
    # deterministically balance the remaining whole-session groups across
    # data seeds. A plain hash modulo can starve one seed when a family has a
    # few very large sessions even though the dataset has ample flow overall.
    session_family_count: Counter[int] = Counter()
    for family in required:
        session_family_count.update(
            np.unique(
                np.concatenate(
                    [
                        merged_by_key[(family, slot)]["session_ids"]
                        for slot in range(n_slots)
                    ]
                )
            ).tolist()
        )
    ambiguous_sessions = np.asarray(
        [
            session
            for session, family_count in session_family_count.items()
            if family_count > 1
        ],
        dtype=np.uint64,
    )
    balanced_by_key: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}
    for family in required:
        merged_family = {
            name: np.concatenate(
                [
                    merged_by_key[(family, slot)][name]
                    for slot in range(n_slots)
                ]
            )
            for name in merged_by_key[(family, 0)]
        }
        if len(ambiguous_sessions):
            keep = ~np.isin(
                merged_family["session_ids"], ambiguous_sessions
            )
            merged_family = {
                name: values[keep]
                for name, values in merged_family.items()
            }
        sessions, inverse = np.unique(
            merged_family["session_ids"], return_inverse=True
        )
        sizes = np.bincount(inverse, minlength=len(sessions))
        tiebreak = _splitmix64(sessions ^ HASH_SALT)
        session_order = np.lexsort((tiebreak, -sizes))
        loads = np.zeros(n_slots, dtype=np.int64)
        session_slot = np.empty(len(sessions), dtype=np.int16)
        for session_index in session_order:
            slot = int(np.argmin(loads))
            session_slot[int(session_index)] = slot
            loads[slot] += int(sizes[int(session_index)])
        target_load = (
            stream_take_by_family[family]
            + design_take_per_slot[family]
        )
        # LPT is close to balanced, but an indivisible high-volume session
        # can still leave one slot just below quota. Move only whole sessions
        # from slots that remain above quota after the move.
        while np.any(loads < target_load):
            receiver = int(np.argmin(loads))
            candidates: list[tuple[int, int, int]] = []
            for session_index, donor in enumerate(session_slot.tolist()):
                if donor == receiver:
                    continue
                size = int(sizes[session_index])
                if loads[donor] - size < target_load:
                    continue
                imbalance = abs(
                    int(loads[receiver] + size)
                    - int(loads[donor] - size)
                )
                candidates.append((imbalance, size, session_index))
            if not candidates:
                break
            _, size, session_index = min(candidates)
            donor = int(session_slot[session_index])
            session_slot[session_index] = receiver
            loads[donor] -= size
            loads[receiver] += size
        row_slots = session_slot[inverse]
        for slot in range(n_slots):
            indexes = np.flatnonzero(row_slots == slot)
            balanced_by_key[(family, slot)] = {
                name: values[indexes]
                for name, values in merged_family.items()
            }
    merged_by_key = balanced_by_key

    for key, merged in merged_by_key.items():
        family, slot = key
        order = np.argsort(merged["rank"], kind="stable")
        stream_take = stream_take_by_family[family]
        design_take = design_take_per_slot[family]
        if (
            spec.name == "NF-UNSW-NB15-v3"
            and family == "generic"
            and design_take == 0
        ):
            # Generic contains a few very large directional 5-tuples. Retain
            # the complete session-filtered seed pool so later segment
            # partitioning can keep those groups intact without starving the
            # opposite segment. The stream still draws exactly the recipe
            # quota without replacement.
            stream_take = len(order)
        take = stream_take + design_take
        if len(order) < take:
            raise ValueError(
                f"{spec.name}/{family}/{key[1]}: collected {len(order)}, need {take}"
            )
        if len(order) < take:
            raise ValueError(
                f"{spec.name}/{family}/{key[1]}: collected {len(order)}, "
                f"need {take}"
            )
        selected = order[:stream_take]
        stream_data = {
            name: values_array[selected]
            for name, values_array in merged.items()
            if name != "rank"
        }
        pools[key] = stream_data
        if family in s5_candidate_required:
            candidate_pools[key] = stream_data
        if design_take:
            design_selected = order[
                stream_take : stream_take + design_take
            ]
            for name in design_parts[family]:
                design_parts[family][name].append(
                    merged[name][design_selected]
                )
    design_pool: dict[str, dict[str, np.ndarray]] = {}
    for family, values in design_parts.items():
        needed = s5_design_required[family]
        merged = {
            name: np.concatenate(parts) for name, parts in values.items()
        }
        if len(merged["x"]) < needed:
            raise ValueError(
                f"{spec.name}/{family}: S5 design pool has "
                f"{len(merged['x'])}, needs {needed}"
            )
        design_pool[family] = {
            name: array[:needed] for name, array in merged.items()
        }
    return (
        feature_columns,
        pools,
        candidate_pools,
        design_pool,
        observed,
    )


def _build_stream(
    spec: DatasetSpec,
    feature_columns: list[str],
    pools: dict[tuple[str, int], dict[str, np.ndarray]],
    s5_candidate_pools: dict[
        tuple[str, int], dict[str, np.ndarray]
    ],
    s5_probes: S5ProbeBundle | None,
    slot: int,
    stream_seed: int,
    scenario: str,
    window_size: int,
    n_windows: int,
    warmup_windows: int,
    local_fraction: float,
    s5_time_quantiles: int,
    s5_min_scope: float,
    s5_min_recall_drop: float,
    s5_min_macro_f1_drop: float,
    s5_min_scope_uplift: float,
    s5_min_recall_drop_uplift: float,
    s5_min_macro_f1_drop_uplift: float,
    s5_gate_policy: str,
) -> tuple[dict[str, np.ndarray], dict[str, object] | None]:
    attack_count = int(round(window_size * spec.attack_rate))
    benign_count = window_size - attack_count
    cursors: Counter[tuple[str, int]] = Counter()
    rng = np.random.default_rng(stream_seed * 1_000_003 + sum(map(ord, scenario)))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subtypes: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    session_ids: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    evasiveness_scores: list[np.ndarray] = []
    matched_control_xs: list[np.ndarray] = []
    phases: list[str] = []
    scenario_pools = pools
    s5_control_pools: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}
    control_cursors: Counter[tuple[str, int]] = Counter()
    if scenario in STRESS_SCENARIOS:
        if s5_probes is None:
            raise ValueError("S5 requires a fitted probe bundle")
        scenario_pools = dict(pools)
        s5_required = _scenario_attack_requirements(
            spec,
            scenario,
            window_size,
            n_windows,
            local_fraction,
            start_window=12,
        )
        for family, count in s5_required.items():
            candidate_pool = s5_candidate_pools[(family, slot)]
            scores = s5_probes.benign_scores(candidate_pool["x"])
            selected = _select_top_by_time_quantile(
                candidate_pool,
                scores,
                count,
                s5_time_quantiles,
                rng,
            )
            control = _select_matched_random_by_time_quantile(
                candidate_pool,
                count,
                s5_time_quantiles,
                selected["source_ids"],
                rng,
            )
            scenario_pools[(family, slot)] = selected
            s5_control_pools[(family, slot)] = control

    scenario_salt = np.uint64(
        sum((index + 1) * ord(value) for index, value in enumerate(scenario))
    )
    segment_requirements: list[Counter[str]] = [
        Counter() for _ in range(4)
    ]
    for planned_window in range(n_windows):
        if planned_window < warmup_windows:
            planned_segment = 0
        elif planned_window < 12:
            planned_segment = 1
        elif planned_window < 24:
            planned_segment = 2
        else:
            planned_segment = 3
        planned_recipe, _ = _window_recipe(
            spec,
            scenario,
            planned_window,
            attack_count,
            local_fraction,
        )
        segment_requirements[planned_segment].update(planned_recipe)
        segment_requirements[planned_segment]["benign"] += benign_count

    segment_probabilities: dict[str, np.ndarray] = {}
    for family, _ in scenario_pools:
        counts = np.asarray(
            [
                segment_requirements[index].get(family, 0)
                for index in range(4)
            ],
            dtype=float,
        )
        if counts.sum() > 0:
            segment_probabilities[family] = counts / counts.sum()

    def partition_pool(
        family: str,
        pool: dict[str, np.ndarray],
    ) -> dict[int, dict[str, np.ndarray]]:
        session_values = pool.get("session_ids", pool["source_ids"]).astype(
            np.uint64
        )
        unique_sessions, inverse = np.unique(
            session_values, return_inverse=True
        )
        row_order = np.argsort(inverse, kind="stable")
        boundaries = np.flatnonzero(np.diff(inverse[row_order])) + 1
        grouped_rows = np.split(row_order, boundaries)
        session_sizes = np.bincount(
            inverse, minlength=len(unique_sessions)
        )
        session_tiebreak = _splitmix64(
            unique_sessions ^ scenario_salt
        )
        # Largest-processing-time-first quota balancing prevents one
        # high-volume 5-tuple from being encountered late and starving a
        # segment. The hash remains a deterministic tie-breaker.
        session_order = np.lexsort(
            (session_tiebreak, -session_sizes)
        )
        targets = segment_probabilities[family] * len(session_values)
        active = targets > 0
        assigned_counts = np.zeros(4, dtype=np.int64)
        assigned_rows: list[list[np.ndarray]] = [
            [] for _ in range(4)
        ]
        for session_index in session_order:
            candidates = np.flatnonzero(active)
            ratios = assigned_counts[candidates] / targets[candidates]
            segment = int(candidates[np.argmin(ratios)])
            rows = grouped_rows[int(session_index)]
            assigned_rows[segment].append(rows)
            assigned_counts[segment] += len(rows)
        result: dict[int, dict[str, np.ndarray]] = {}
        for segment in range(4):
            indexes = (
                np.concatenate(assigned_rows[segment])
                if assigned_rows[segment]
                else np.empty(0, dtype=np.int64)
            )
            required_count = int(
                segment_requirements[segment].get(family, 0)
            )
            if len(indexes) < required_count:
                raise ValueError(
                    f"{spec.name}/{scenario}/{stream_seed}/{family}/"
                    f"segment{segment}: quota-balanced session partition "
                    f"has {len(indexes)}, needs {required_count}"
                )
            result[segment] = {
                name: values[indexes] for name, values in pool.items()
            }
        return result

    segmented_pools: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}
    for (family, pool_slot), pool in scenario_pools.items():
        if pool_slot != slot or family not in segment_probabilities:
            continue
        for segment, values in partition_pool(family, pool).items():
            segmented_pools[(family, segment)] = values
    segmented_control_pools: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}
    for (family, pool_slot), pool in s5_control_pools.items():
        if pool_slot != slot or family not in segment_probabilities:
            continue
        for segment, values in partition_pool(family, pool).items():
            segmented_control_pools[(family, segment)] = values

    def draw(
        family: str, count: int, segment: int
    ) -> dict[str, np.ndarray]:
        key = (family, segment)
        start = cursors[key]
        stop = start + int(count)
        pool = segmented_pools[key]
        if stop > len(pool["x"]):
            raise ValueError(
                f"{spec.name}/{scenario}/{stream_seed}/{family}/"
                f"segment{segment}: "
                f"requested {stop}, available {len(pool['x'])}"
            )
        cursors[key] = stop
        return {name: values[start:stop] for name, values in pool.items()}

    def draw_control(
        family: str, count: int, segment: int
    ) -> dict[str, np.ndarray]:
        key = (family, segment)
        start = control_cursors[key]
        stop = start + int(count)
        pool = segmented_control_pools[key]
        if stop > len(pool["x"]):
            raise ValueError(
                f"{spec.name}/{scenario}/{stream_seed}/{family}: matched control "
                f"requested {stop}, available {len(pool['x'])}"
            )
        control_cursors[key] = stop
        return {name: values[start:stop] for name, values in pool.items()}

    for window in range(n_windows):
        if window < warmup_windows:
            segment = 0
        elif window < 12:
            segment = 1
        elif window < 24:
            segment = 2
        else:
            segment = 3
        recipe, phase = _window_recipe(
            spec, scenario, window, attack_count, local_fraction
        )
        parts: list[dict[str, np.ndarray]] = []
        part_labels: list[np.ndarray] = []
        part_subtypes: list[np.ndarray] = []
        control_parts: list[dict[str, np.ndarray]] = []
        benign = draw("benign", benign_count, segment)
        parts.append(benign)
        control_parts.append(benign)
        part_labels.append(np.zeros(benign_count, dtype=np.int64))
        part_subtypes.append(np.full(benign_count, "benign", dtype="<U48"))
        for family, count in recipe.items():
            attack = draw(family, count, segment)
            parts.append(attack)
            if scenario in STRESS_SCENARIOS and window >= 12:
                control_parts.append(
                    draw_control(family, count, segment)
                )
            else:
                control_parts.append(attack)
            part_labels.append(np.ones(count, dtype=np.int64))
            part_subtypes.append(np.full(count, family, dtype="<U48"))
        window_x = np.concatenate([part["x"] for part in parts])
        matched_control_window_x = np.concatenate(
            [part["x"] for part in control_parts]
        )
        window_y = np.concatenate(part_labels)
        window_subtypes = np.concatenate(part_subtypes)
        window_ids = np.concatenate([part["source_ids"] for part in parts])
        window_sessions = np.concatenate(
            [
                part.get("session_ids", part["source_ids"])
                for part in parts
            ]
        )
        window_times = np.concatenate([part["timestamps"] for part in parts])
        window_scores = np.concatenate(
            [
                part.get(
                    "evasive_score",
                    np.full(len(part["x"]), np.nan, dtype=np.float32),
                )
                for part in parts
            ]
        )
        order = rng.permutation(window_size)
        xs.append(window_x[order])
        matched_control_xs.append(matched_control_window_x[order])
        ys.append(window_y[order])
        subtypes.append(window_subtypes[order])
        source_ids.append(window_ids[order])
        session_ids.append(window_sessions[order])
        timestamps.append(window_times[order])
        evasiveness_scores.append(window_scores[order])
        phases.append("warmup" if window < warmup_windows else phase)

    raw_x = np.stack(xs)
    matched_control_raw_x = np.stack(matched_control_xs)
    labels = np.stack(ys)
    ids = np.stack(source_ids)
    sessions = np.stack(session_ids)
    if len(np.unique(ids)) != ids.size:
        raise ValueError(
            f"{spec.name}/{scenario}/{stream_seed}: source row overlap detected"
        )
    segment_slices = [
        sessions[:warmup_windows],
        sessions[warmup_windows:12],
        sessions[12:24],
        sessions[24:36],
    ]
    segment_sets = [
        set(values.reshape(-1).tolist()) for values in segment_slices
    ]
    for left_index, left in enumerate(segment_sets):
        for right in segment_sets[left_index + 1 :]:
            if left.intersection(right):
                raise ValueError(
                    f"{spec.name}/{scenario}/{stream_seed}: directional "
                    "5-tuple overlap detected across stream segments"
                )
    observed_attack = labels.mean(axis=1)
    expected_attack = attack_count / window_size
    if not np.allclose(observed_attack, expected_attack, atol=1 / window_size):
        raise ValueError("Per-window attack rate deviates from the locked value")

    s5_audit: dict[str, object] | None = None
    if scenario in STRESS_SCENARIOS:
        assert s5_probes is not None
        s5_audit = _s5_audit(
            raw_x,
            matched_control_raw_x,
            labels,
            s5_probes.audit_models,
            s5_min_scope,
            s5_min_recall_drop,
            s5_min_macro_f1_drop,
            s5_min_scope_uplift,
            s5_min_recall_drop_uplift,
            s5_min_macro_f1_drop_uplift,
        )
        if not s5_audit["passed"]:
            message = (
                f"{spec.name}/{scenario}/{stream_seed}: "
                "high-severity gates failed: "
                f"{s5_audit['conditions']}"
            )
            if s5_gate_policy == "enforce":
                raise ValueError(message)
            if s5_gate_policy == "warn":
                print(f"WARNING {message}", flush=True)

    warmup = raw_x[:warmup_windows].reshape(-1, raw_x.shape[-1])
    medians = np.nanmedian(warmup, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(raw_x), raw_x, medians)
    scaler = StandardScaler().fit(
        imputed[:warmup_windows].reshape(-1, imputed.shape[-1])
    )
    scaled = scaler.transform(imputed.reshape(-1, imputed.shape[-1])).reshape(
        imputed.shape
    )
    arrays = {
        "x": scaled.astype(np.float32),
        "y": labels.astype(np.int64),
        "subtypes": np.stack(subtypes).astype("<U48"),
        "source_ids": ids.astype(np.int64),
        "session_ids": sessions.astype(np.uint64),
        "timestamps": np.stack(timestamps).astype(np.int64),
        "phases": np.asarray(phases, dtype="<U24"),
        "feature_names": np.asarray(feature_columns, dtype="<U64"),
        "class_names": np.asarray(["Benign", "Attack"], dtype="<U16"),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "imputation_median": medians.astype(np.float64),
    }
    if scenario in STRESS_SCENARIOS:
        arrays["s5_evasiveness_score"] = np.stack(
            evasiveness_scores
        ).astype(np.float32)
    return arrays, s5_audit


def _write_stream(
    destination: Path,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, object],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Final matrices favor deterministic build throughput over archive size.
    # The NPZ container and array content are unchanged; only compression is
    # disabled.
    np.savez(destination, **arrays)
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-audited controlled NF-v3 stream artifacts."
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset keys.",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(CANONICAL_SCENARIOS),
        help=(
            "Comma-separated scenarios. S8-Stress is an explicit high-severity "
            "stress track and is not included in the canonical default."
        ),
    )
    parser.add_argument("--output-dir", default="prepared_streams")
    parser.add_argument("--stream-seeds", default="2101,2102,2103,2104,2105")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--n-windows", type=int, default=36)
    parser.add_argument("--warmup-windows", type=int, default=4)
    parser.add_argument("--local-fraction", type=float, default=0.05)
    parser.add_argument(
        "--s5-candidate-multiplier", type=float, default=3.0
    )
    parser.add_argument("--s5-design-per-class", type=int, default=4096)
    parser.add_argument("--s5-time-quantiles", type=int, default=4)
    parser.add_argument("--s5-probe-seed", type=int, default=20260729)
    parser.add_argument(
        "--s5-min-scope",
        type=float,
        default=0.70,
        help=(
            "Minimum fraction of post-drift attacks classified as benign "
            "by a majority of the independent audit probes."
        ),
    )
    parser.add_argument(
        "--s5-min-attack-recall-drop", type=float, default=0.30
    )
    parser.add_argument(
        "--s5-min-macro-f1-drop", type=float, default=0.20
    )
    parser.add_argument(
        "--s5-min-evasion-uplift-vs-random", type=float, default=0.10
    )
    parser.add_argument(
        "--s5-min-attack-recall-drop-uplift-vs-random",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--s5-min-macro-f1-drop-uplift-vs-random",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--s5-gate-policy",
        choices=["enforce", "warn", "off"],
        default="enforce",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--exclude-design-fold-modulus",
        type=int,
        default=0,
        help=(
            "If >1, exclude sessions whose deterministic construct-design "
            "fold equals --exclude-design-fold-residue."
        ),
    )
    parser.add_argument(
        "--exclude-design-fold-residue", type=int, default=0
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_keys = [value.strip() for value in args.datasets.split(",") if value.strip()]
    unknown = set(dataset_keys).difference(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    selected_scenarios = tuple(
        value.strip()
        for value in args.scenarios.split(",")
        if value.strip()
    )
    unknown_scenarios = set(selected_scenarios).difference(SCENARIOS)
    if unknown_scenarios:
        raise ValueError(
            f"Unknown scenarios: {sorted(unknown_scenarios)}"
        )
    if not selected_scenarios:
        raise ValueError("At least one scenario is required")
    stream_seeds = [
        int(value.strip())
        for value in args.stream_seeds.split(",")
        if value.strip()
    ]
    if len(stream_seeds) != len(set(stream_seeds)):
        raise ValueError("stream seeds must be unique")
    if not 0.0 < args.local_fraction < 1.0:
        raise ValueError("local_fraction must be in (0, 1)")
    if args.s5_candidate_multiplier < 2.0:
        raise ValueError(
            "s5-candidate-multiplier must be >= 2 because S5 requires an "
            "equally sized, source-disjoint matched-random control"
        )
    if args.exclude_design_fold_modulus < 0:
        raise ValueError("exclude-design-fold-modulus must be non-negative")
    if (
        args.exclude_design_fold_modulus > 1
        and not 0
        <= args.exclude_design_fold_residue
        < args.exclude_design_fold_modulus
    ):
        raise ValueError(
            "exclude-design-fold-residue must be within the modulus"
        )
    if args.s5_design_per_class < 32:
        raise ValueError("s5-design-per-class must be >= 32")
    if args.s5_time_quantiles < 1:
        raise ValueError("s5-time-quantiles must be >= 1")
    for name, value in (
        ("s5-min-scope", args.s5_min_scope),
        ("s5-min-attack-recall-drop", args.s5_min_attack_recall_drop),
        ("s5-min-macro-f1-drop", args.s5_min_macro_f1_drop),
        (
            "s5-min-evasion-uplift-vs-random",
            args.s5_min_evasion_uplift_vs_random,
        ),
        (
            "s5-min-attack-recall-drop-uplift-vs-random",
            args.s5_min_attack_recall_drop_uplift_vs_random,
        ),
        (
            "s5-min-macro-f1-drop-uplift-vs-random",
            args.s5_min_macro_f1_drop_uplift_vs_random,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    output_root = Path(args.output_dir)
    for dataset_key in dataset_keys:
        spec = DATASETS[dataset_key]
        expected_paths = [
            output_root
            / dataset_key
            / scenario
            / f"stream_seed{stream_seed}.npz"
            for scenario in selected_scenarios
            for stream_seed in stream_seeds
        ]
        if args.resume and all(
            path.exists() and path.with_suffix(".manifest.json").exists()
            for path in expected_paths
        ):
            print(f"resume dataset={spec.name}: all artifacts exist", flush=True)
            continue
        attack_count, benign_count, required = _requirements(
            spec,
            selected_scenarios,
            args.window_size,
            args.n_windows,
            args.local_fraction,
        )
        selected_stress = [
            value
            for value in selected_scenarios
            if value in STRESS_SCENARIOS
        ]
        s5_enabled = bool(selected_stress)
        s5_candidate_required = (
            _scenario_attack_requirements(
                spec,
                selected_stress[0],
                args.window_size,
                args.n_windows,
                args.local_fraction,
                start_window=12,
            )
            if s5_enabled
            else {}
        )
        s5_design_required = (
            {
                "benign": 2 * args.s5_design_per_class,
                **dict(
                    Counter(
                        _allocate(
                            args.s5_design_per_class,
                            spec.concept_a,
                        )
                    )
                    + Counter(
                        _allocate(
                            args.s5_design_per_class,
                            spec.concept_a,
                        )
                    )
                ),
            }
            if s5_enabled
            else {}
        )
        source_path = Path(spec.path)
        source_sha256 = _sha256(source_path)
        (
            feature_columns,
            pools,
            s5_candidate_pools,
            s5_design_pool,
            observed,
        ) = _collect_pools(
            spec,
            stream_seeds,
            required,
            s5_candidate_required,
            s5_design_required,
            args.s5_candidate_multiplier,
            args.chunksize,
            args.exclude_design_fold_modulus,
            args.exclude_design_fold_residue,
        )
        if s5_enabled:
            design_ids = set(
                np.concatenate(
                    [
                        values["source_ids"]
                        for values in s5_design_pool.values()
                    ]
                ).tolist()
            )
            stream_candidate_ids = set(
                np.concatenate(
                    [
                        values["source_ids"]
                        for values in pools.values()
                    ]
                    + [
                        values["source_ids"]
                        for values in s5_candidate_pools.values()
                    ]
                ).tolist()
            )
            design_overlap = design_ids.intersection(stream_candidate_ids)
            if design_overlap:
                raise ValueError(
                    f"{spec.name}: S5 design/stream pool overlap "
                    f"({len(design_overlap)} rows)"
                )
        s5_probes = (
            _fit_s5_probes(
                spec,
                s5_design_pool,
                args.s5_design_per_class,
                args.s5_probe_seed,
            )
            if s5_enabled
            else None
        )
        seed_id_sets: dict[int, set[int]] = {}
        seed_session_sets: dict[int, set[int]] = {}
        for slot, stream_seed in enumerate(stream_seeds):
            seed_id_sets[stream_seed] = set(
                np.concatenate(
                    [
                        pools[(family, slot)]["source_ids"]
                        for family in required
                    ]
                    + [
                        s5_candidate_pools[(family, slot)][
                            "source_ids"
                        ]
                        for family in s5_candidate_required
                    ]
                ).tolist()
            )
            seed_session_sets[stream_seed] = set(
                np.concatenate(
                    [
                        pools[(family, slot)]["session_ids"]
                        for family in required
                    ]
                    + [
                        s5_candidate_pools[(family, slot)][
                            "session_ids"
                        ]
                        for family in s5_candidate_required
                    ]
                ).tolist()
            )
        for left_index, left_seed in enumerate(stream_seeds):
            for right_seed in stream_seeds[left_index + 1 :]:
                overlap = seed_id_sets[left_seed].intersection(seed_id_sets[right_seed])
                if overlap:
                    raise ValueError(
                        f"{spec.name}: data seeds {left_seed}/{right_seed} overlap"
                    )
                session_overlap = seed_session_sets[
                    left_seed
                ].intersection(seed_session_sets[right_seed])
                if session_overlap:
                    raise ValueError(
                        f"{spec.name}: data seeds {left_seed}/{right_seed} "
                        f"overlap on {len(session_overlap)} directional "
                        "5-tuples"
                    )

        for slot, stream_seed in enumerate(stream_seeds):
            for scenario in selected_scenarios:
                destination = (
                    output_root
                    / dataset_key
                    / scenario
                    / f"stream_seed{stream_seed}.npz"
                )
                arrays, s5_audit = _build_stream(
                    spec,
                    feature_columns,
                    pools,
                    s5_candidate_pools,
                    s5_probes,
                    slot,
                    stream_seed,
                    scenario,
                    args.window_size,
                    args.n_windows,
                    args.warmup_windows,
                    args.local_fraction,
                    args.s5_time_quantiles,
                    args.s5_min_scope,
                    args.s5_min_attack_recall_drop,
                    args.s5_min_macro_f1_drop,
                    args.s5_min_evasion_uplift_vs_random,
                    args.s5_min_attack_recall_drop_uplift_vs_random,
                    args.s5_min_macro_f1_drop_uplift_vs_random,
                    args.s5_gate_policy,
                )
                subtype_counts = Counter(arrays["subtypes"].reshape(-1).tolist())
                manifest = {
                    "protocol_version": "1.1",
                    "dataset_key": dataset_key,
                    "dataset_name": spec.name,
                    "source_path": str(source_path.resolve()),
                    "source_sha256": source_sha256,
                    "source_rows": spec.total_rows,
                    "observed_required_family_counts": dict(observed),
                    "scenario": scenario,
                    "stream_seed": stream_seed,
                    "window_size": args.window_size,
                    "n_windows": args.n_windows,
                    "warmup_windows": args.warmup_windows,
                    "attack_count_per_window": attack_count,
                    "benign_count_per_window": benign_count,
                    "realized_attack_rate": attack_count / args.window_size,
                    "local_fraction_of_attacks": LOCAL_LEVELS.get(
                        scenario, args.local_fraction
                    ),
                    "legacy_cli_local_fraction": args.local_fraction,
                    "feature_count": len(feature_columns),
                    "subtype_counts": dict(subtype_counts),
                    "unique_source_rows": int(
                        len(np.unique(arrays["source_ids"]))
                    ),
                    "overlap_within_stream": 0,
                    "session_overlap_within_stream": 0,
                    "cross_seed_overlap": 0,
                    "session_proxy": (
                        "directional 5-tuple x fixed 30-minute epoch bucket"
                        if spec.name == "NF-BoT-IoT-v3"
                        else "directional 5-tuple over the capture"
                    ),
                    "seed_assignment": {
                        "method": (
                            "whole-directional-session deterministic "
                            "load balancing"
                        ),
                        "salt": int(HASH_SALT),
                        "selection_basis": (
                            "session-group size with label-blind hash "
                            "tie-break; no model outcome"
                        ),
                    },
                    "excluded_construct_design_fold": (
                        {
                            "modulus": args.exclude_design_fold_modulus,
                            "residue": args.exclude_design_fold_residue,
                            "salt": int(DESIGN_SALT),
                        }
                        if args.exclude_design_fold_modulus > 1
                        else None
                    ),
                    "preprocessing_fit_windows": list(
                        range(args.warmup_windows)
                    ),
                    "concept_definitions": {
                        "A": spec.concept_a,
                        "B": spec.concept_b,
                        "C": spec.concept_c,
                        "A_prime": spec.concept_a,
                        "local_stable": spec.stable,
                        "local_A": spec.local_a,
                        "local_B": spec.local_b,
                    },
                }
                if scenario in STRESS_SCENARIOS:
                    finite_scores = arrays["s5_evasiveness_score"][
                        np.isfinite(arrays["s5_evasiveness_score"])
                    ]
                    manifest["s5"] = {
                        "name": (
                            "high-severity adversarial "
                            "class-conditional drift"
                        ),
                        "candidate_multiplier": (
                            args.s5_candidate_multiplier
                        ),
                        "time_quantiles": args.s5_time_quantiles,
                        "probe_seed": args.s5_probe_seed,
                        "selection_models": [
                            "logistic_regression",
                            "random_forest",
                            "extra_trees",
                        ],
                        "design_per_class_per_split": (
                            args.s5_design_per_class
                        ),
                        "design_stream_overlap": 0,
                        "matched_random_control": {
                            "same_family_weights": True,
                            "same_time_quantiles": True,
                            "same_attack_prevalence": True,
                            "source_disjoint_from_selected": True,
                        },
                        "selected_evasive_rows": int(
                            len(finite_scores)
                        ),
                        "evasive_score_mean": float(
                            np.mean(finite_scores)
                        ),
                        "evasive_score_median": float(
                            np.median(finite_scores)
                        ),
                        "evasive_score_min": float(
                            np.min(finite_scores)
                        ),
                        "audit": s5_audit,
                        "gate_policy": args.s5_gate_policy,
                    }
                _write_stream(destination, arrays, manifest)
                print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
