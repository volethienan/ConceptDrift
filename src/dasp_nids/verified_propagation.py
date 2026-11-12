"""Same-window incident verification under label delay.

Bulk labels arrive ``label_delay`` windows late, but an analyst can verify a
few flows of the current window at once (the fast lane). Attack bursts are
homogeneous, so one verified flow can stand for a whole group of flows. The
functions here spend a fast-label budget on the flows the model currently
calls benign and return the flows to raise as attack in the same window.

Every function reads labels only through ``LabelOracle``, which records each
query and enforces the budget; no other label of the window is reachable.

* ``purity_verified_propagation``: clusters are verified with ``r`` labels
  drawn uniformly at random from the cluster. An attack verdict is spread to
  the whole cluster only when the one-sided Clopper-Pearson lower bound on
  its attack fraction is at least ``1 - epsilon`` (level ``delta``); a
  cluster with mixed labels is split and its children are verified with
  fresh draws. A cluster passing the test holds at most an ``epsilon``
  benign fraction with probability at least ``1 - delta``.
* ``representative_propagation``: one label per cluster, nearest to the
  centroid, spread without a purity test (the pilot "alarm-then-verify").
* ``uniform_knn_propagation``: uniform fast labels; each candidate takes the
  label of its nearest verified flow.
* ``hierarchical_sampling``: Dasgupta and Hsu (ICML 2008) on a Ward tree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.stats import beta
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier


class LabelBudgetExceeded(RuntimeError):
    pass


class LabelOracle:
    """Reveals labels of single flows, records each query, enforces a budget.

    ``noise`` models analyst mistakes: each flow's revealed label is flipped
    with that probability (fixed per flow, so a repeated query agrees).
    """

    def __init__(self, labels: np.ndarray, budget: int, noise: float = 0.0,
                 rng: np.random.Generator | None = None):
        labels = np.asarray(labels).astype(np.int64)
        if noise > 0:
            flips = (rng or np.random.default_rng(0)).random(len(labels)) < noise
            labels = np.where(flips, 1 - labels, labels)
        self._labels = labels
        self.budget = int(budget)
        self.queried: list[int] = []

    @property
    def remaining(self) -> int:
        return self.budget - len(self.queried)

    def query(self, indices) -> np.ndarray:
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        fresh = [int(i) for i in indices if int(i) not in self._seen]
        if len(fresh) > self.remaining:
            raise LabelBudgetExceeded(f"{len(fresh)} new queries, {self.remaining} left")
        self.queried.extend(fresh)
        return self._labels[indices].astype(np.int64)

    @property
    def _seen(self) -> set[int]:
        return set(self.queried)


@dataclass
class Verification:
    """Fast-lane outcome for one window."""

    queried: np.ndarray
    alarms: np.ndarray  # flows raised as attack (queried attacks included)
    pseudo_indices: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    pseudo_labels: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    diagnostics: dict = field(default_factory=dict)
    # groups of flows alarmed as one verified unit (certified clusters)
    groups: list = field(default_factory=list)


def group_signatures(z: np.ndarray, groups: list, quantile: float = 0.9) -> list[tuple[np.ndarray, float]]:
    """(centroid, radius) of each alarmed group in cluster space; the radius is
    the ``quantile`` of member distances to the centroid."""
    out = []
    for members in groups:
        if not len(members):
            continue
        centroid = z[members].mean(axis=0)
        radius = float(np.quantile(np.linalg.norm(z[members] - centroid, axis=1), quantile))
        out.append((centroid, radius))
    return out


def match_signatures(z: np.ndarray, candidates: np.ndarray, signatures: list) -> np.ndarray:
    """Candidates inside the radius of any signature (verified on an earlier window)."""
    if not len(candidates) or not signatures:
        return np.empty(0, np.int64)
    inside = np.zeros(len(candidates), dtype=bool)
    for centroid, radius in signatures:
        inside |= np.linalg.norm(z[candidates] - centroid, axis=1) <= radius
    return candidates[inside].astype(np.int64)


class AnomalyAlarm:
    """Label-free baseline: Isolation Forest fitted once on reference benign
    flows (labels already revealed); alarms candidates scoring above the
    ``1 - alpha`` quantile of the reference scores."""

    def __init__(self, reference: np.ndarray, alpha: float = 0.01, seed: int = 0):
        from sklearn.ensemble import IsolationForest

        z = cluster_space(reference)
        self.forest = IsolationForest(n_estimators=100, random_state=seed).fit(z)
        self.threshold = float(np.quantile(-self.forest.score_samples(z), 1 - alpha))

    def alarms(self, x: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        if not len(candidates):
            return np.empty(0, np.int64)
        score = -self.forest.score_samples(cluster_space(x[candidates]))
        return candidates[score > self.threshold].astype(np.int64)


def cluster_space(x: np.ndarray) -> np.ndarray:
    """Signed log scale: the z-scored flow features are heavy tailed."""
    return np.sign(x) * np.log1p(np.abs(x))


def clopper_pearson_lower(successes: int, trials: int, delta: float) -> float:
    if trials == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(delta, successes, trials - successes + 1))


def novelty_candidates(reference: np.ndarray, x: np.ndarray, model_benign: np.ndarray,
                       alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Model-benign flows that a reference-benign-vs-now classifier flags (cross-fitted).

    ``reference`` holds flows of earlier windows whose labels have arrived and
    are benign. The threshold is the ``1 - alpha`` quantile of held-out
    reference scores, so no label of the current window is used.
    """
    order = rng.permutation(len(reference))
    fit, calibration = reference[order[: len(order) // 2]], reference[order[len(order) // 2:]]
    halves = np.array_split(rng.permutation(len(x)), 2)
    score, calibration_scores = np.zeros(len(x)), []
    for train_half, score_half in ((halves[0], halves[1]), (halves[1], halves[0])):
        features = np.concatenate([fit, x[train_half]])
        target = np.r_[np.zeros(len(fit)), np.ones(len(train_half))]
        weight = np.where(target == 1, len(fit) / len(train_half), 1.0)
        classifier = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=0)
        classifier.fit(features, target, sample_weight=weight)
        score[score_half] = classifier.predict_proba(x[score_half])[:, 1]
        calibration_scores.append(classifier.predict_proba(calibration)[:, 1])
    threshold = np.quantile(np.mean(calibration_scores, axis=0), 1 - alpha)
    return np.flatnonzero((score > threshold) & model_benign)


def _kmeans(z: np.ndarray, k: int, seed: int) -> np.ndarray:
    k = max(1, min(k, len(np.unique(z, axis=0))))
    if k == 1:
        return np.zeros(len(z), dtype=np.int64)
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(z)


CLUSTERERS = ("kmeans", "ward", "random")


def _partition(z: np.ndarray, k: int, seed: int, method: str = "kmeans") -> np.ndarray:
    """Split ``z`` into at most ``k`` groups: k-means (default), Ward
    agglomeration, or a random partition (ablation: no feature structure)."""
    if method == "kmeans":
        return _kmeans(z, k, seed)
    if method == "ward":
        k = max(1, min(k, len(np.unique(z, axis=0))))
        if k == 1:
            return np.zeros(len(z), dtype=np.int64)
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(z).astype(np.int64)
    if method == "random":
        k = max(1, min(k, len(z)))
        return np.random.default_rng(seed).integers(0, k, size=len(z)).astype(np.int64)
    raise ValueError(f"unknown clusterer {method}; expected one of {CLUSTERERS}")


def feature_space(x: np.ndarray, space: str = "log") -> np.ndarray:
    """Clustering space: signed log (default) or the raw z-scored features."""
    if space == "log":
        return cluster_space(x)
    if space == "raw":
        return np.asarray(x, dtype=float)
    raise ValueError(f"unknown cluster space {space}")


def purity_verified_propagation(x: np.ndarray, candidates: np.ndarray, oracle: LabelOracle,
                                rng: np.random.Generator, *, initial_clusters: int = 8,
                                labels_per_test: int = 4, epsilon: float = 0.3,
                                delta: float = 0.25, min_cluster: int = 10,
                                split_mixed: bool = True, clusterer: str = "kmeans",
                                space: str = "log") -> Verification:
    """Screen clusters of ``candidates`` with one random label, confirm attack ones, spread certified attacks.

    Each cluster draws members uniformly at random without replacement, one
    label at a time. A benign first draw drops the cluster (screening, largest
    clusters first). A cluster whose draws so far are all attack is confirmed
    before any new cluster is screened; once ``labels_per_test`` draws are all
    attack and the Clopper-Pearson lower bound on its attack fraction is at
    least ``1 - epsilon`` (level ``delta``), the whole cluster is alarmed. A
    cluster that shows both labels is split in two (k-means on the members not
    yet drawn) and the children start afresh. Certification needs all of a
    cluster's draws to be attack, so a cluster whose benign fraction exceeds
    ``epsilon`` passes with probability at most ``(1 - epsilon) ** labels_per_test``.

    ``diagnostics["records"]`` lists every cluster that entered the queue with
    its members, the labels drawn and its outcome (``pass``; ``benign``: first
    draw benign; ``mixed``: both labels drawn, split or dropped; ``small``;
    ``open``: the budget ran out first), so the certificate can be audited
    against ground truth outside this function.
    """
    if clopper_pearson_lower(labels_per_test, labels_per_test, delta) < 1 - epsilon:
        raise ValueError("labels_per_test all-attack draws cannot reach 1 - epsilon at level delta")
    z = feature_space(x, space)
    queue: list[dict] = []
    records: list[dict] = []

    def close(cluster: dict, outcome: str) -> None:
        queue[:] = [c for c in queue if c is not cluster]
        records.append({"members": cluster["members"], "labels": list(cluster["labels"]), "outcome": outcome})

    if len(candidates) >= min_cluster:
        assignment = _partition(z[candidates], initial_clusters, int(rng.integers(2**31)), clusterer)
        queue = [{"members": candidates[assignment == c], "drawn": [], "labels": []}
                 for c in np.unique(assignment)]
    certified, tests = [], []
    while queue and oracle.remaining > 0:
        confirming = [c for c in queue if c["labels"]]
        cluster = max(confirming or queue, key=lambda c: len(c["members"]))
        undrawn = np.setdiff1d(cluster["members"], cluster["drawn"])
        if len(cluster["members"]) < min_cluster or not len(undrawn):
            close(cluster, "small")
            continue
        draw = int(rng.choice(undrawn))
        cluster["drawn"].append(draw)
        cluster["labels"].append(int(oracle.query(draw)[0]))
        attacks, n = sum(cluster["labels"]), len(cluster["labels"])
        if attacks == 0:
            close(cluster, "benign")
        elif attacks < n:
            close(cluster, "mixed")
            rest = np.setdiff1d(cluster["members"], cluster["drawn"])
            if split_mixed and len(rest) >= 2 * min_cluster:
                split = _partition(z[rest], 2, int(rng.integers(2**31)), clusterer)
                queue.extend({"members": rest[split == c], "drawn": [], "labels": []} for c in np.unique(split))
        elif n == labels_per_test:
            close(cluster, "pass")
            lower = clopper_pearson_lower(attacks, n, delta)
            tests.append({"size": int(len(cluster["members"])), "lower": lower})
            certified.append(cluster["members"])
    for cluster in list(queue):
        close(cluster, "open")
    queried = np.asarray(oracle.queried, dtype=np.int64)
    queried_attacks = queried[oracle.query(queried) == 1] if len(queried) else queried
    alarms = np.unique(np.concatenate([*certified, queried_attacks])) if certified or len(queried_attacks) else np.empty(0, np.int64)
    propagated = np.concatenate(certified) if certified else np.empty(0, np.int64)
    return Verification(queried, alarms.astype(np.int64), propagated.astype(np.int64),
                        np.ones(len(propagated), dtype=np.int64),
                        diagnostics={"tests": tests, "certified_clusters": len(certified),
                                     "propagated": int(sum(map(len, certified))),
                                     "records": records},
                        groups=list(certified))


def representative_propagation(x: np.ndarray, candidates: np.ndarray, oracle: LabelOracle,
                               rng: np.random.Generator, *, clusters: int = 16) -> Verification:
    """One label per k-means cluster (member nearest the centroid); verdict spread unchecked."""
    if len(candidates) < 2 * clusters:
        return Verification(np.empty(0, np.int64), np.empty(0, np.int64))
    z = cluster_space(x[candidates])
    model = KMeans(n_clusters=clusters, n_init=3, random_state=int(rng.integers(2**31))).fit(z)
    alarms, pseudo_i, pseudo_y = [], [], []
    for c in range(clusters):
        inside = model.labels_ == c
        if not inside.any() or oracle.remaining < 1:
            continue
        rep = candidates[inside][np.argmin(((z[inside] - model.cluster_centers_[c]) ** 2).sum(1))]
        verdict = int(oracle.query(rep)[0])
        pseudo_i.append(candidates[inside]); pseudo_y.append(np.full(inside.sum(), verdict))
        if verdict:
            alarms.append(candidates[inside])
    queried = np.asarray(oracle.queried, dtype=np.int64)
    return Verification(queried, np.unique(np.concatenate(alarms)) if alarms else np.empty(0, np.int64),
                        np.concatenate(pseudo_i) if pseudo_i else np.empty(0, np.int64),
                        np.concatenate(pseudo_y) if pseudo_y else np.empty(0, np.int64),
                        groups=list(alarms))


def uniform_knn_propagation(x: np.ndarray, candidates: np.ndarray, oracle: LabelOracle,
                            rng: np.random.Generator, *, budget: int) -> Verification:
    """Uniform fast labels over the window; each candidate takes its nearest verified flow's label."""
    queried = rng.choice(len(x), size=min(budget, len(x)), replace=False)
    labels = oracle.query(queried)
    if not len(candidates) or not labels.any():
        return Verification(queried, queried[labels == 1])
    z = cluster_space(x)
    nearest = np.argmin(((z[candidates][:, None, :] - z[queried][None, :, :]) ** 2).sum(-1), axis=1)
    alarms = np.union1d(candidates[labels[nearest] == 1], queried[labels == 1])
    return Verification(queried, alarms.astype(np.int64))


def hierarchical_sampling(x: np.ndarray, candidates: np.ndarray, oracle: LabelOracle,
                          rng: np.random.Generator, *, budget: int, beta_admissible: float = 2.0,
                          max_points: int = 4096) -> Verification:
    """Dasgupta & Hsu (2008): sample from a pruning of a Ward tree and refine it.

    Each query draws a node of the current pruning with probability
    proportional to its size, then a uniform random point under it; the
    label counts every node on the path from the drawn node to the point.
    A node is replaced by its children when both children carry an
    admissible label and their weighted error bound is lower. Points take
    the label of their pruning node; only attack labels raise alarms.
    """
    if len(candidates) < 4:
        return Verification(np.empty(0, np.int64), np.empty(0, np.int64))
    points = candidates if len(candidates) <= max_points else rng.choice(candidates, max_points, replace=False)
    n = len(points)
    tree = linkage(cluster_space(x[points]), method="ward")
    children = {n + i: (int(a), int(b)) for i, (a, b, _, _) in enumerate(tree)}
    leaves: dict[int, np.ndarray] = {i: np.array([i]) for i in range(n)}
    for node in range(n, 2 * n - 1):
        a, b = children[node]
        leaves[node] = np.concatenate([leaves[a], leaves[b]])
    parent = {child: node for node, pair in children.items() for child in pair}
    root = 2 * n - 2
    counts = {node: np.zeros(2) for node in leaves}

    def bounds(node: int) -> tuple[np.ndarray, np.ndarray]:
        total = counts[node].sum()
        if total == 0:
            return np.zeros(2), np.ones(2)
        p = counts[node] / total
        width = np.sqrt(1.0 / total)  # DH08 uses p +- c/sqrt(n)
        return np.clip(p - width, 0, 1), np.clip(p + width, 0, 1)

    def admissible(node: int) -> int | None:
        lower, upper = bounds(node)
        for label in (0, 1):
            if lower[label] > beta_admissible * upper[1 - label] - 1:
                return label
        return None

    def error(node: int) -> float:
        label = admissible(node)
        return 1.0 if label is None else 1.0 - counts[node][label] / max(1.0, counts[node].sum())

    pruning = [root]
    sampled = 0
    while sampled < budget and oracle.remaining > 0:
        weights = np.array([len(leaves[v]) for v in pruning], dtype=float)
        node = pruning[rng.choice(len(pruning), p=weights / weights.sum())]
        point = int(rng.choice(leaves[node]))
        label = int(oracle.query(points[point])[0])
        sampled += 1
        walk = point
        while True:
            counts[walk][label] += 1
            if walk == node:
                break
            walk = parent[walk]
        changed = True
        while changed:
            changed = False
            for v in list(pruning):
                if v not in children:
                    continue
                a, b = children[v]
                if admissible(a) is None or admissible(b) is None:
                    continue
                split_error = (len(leaves[a]) * error(a) + len(leaves[b]) * error(b)) / len(leaves[v])
                if split_error < error(v):
                    pruning.remove(v); pruning.extend([a, b]); changed = True
    alarms = []
    for v in pruning:
        label = admissible(v)
        if label is None and counts[v].sum():
            label = int(np.argmax(counts[v]))
        if label == 1:
            alarms.append(points[leaves[v]])
    queried = np.asarray(oracle.queried, dtype=np.int64)
    queried_attacks = queried[oracle.query(queried) == 1] if len(queried) else queried
    pieces = [*alarms, queried_attacks]
    return Verification(queried, np.unique(np.concatenate(pieces)).astype(np.int64))
