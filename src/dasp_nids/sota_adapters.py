"""Budget-matched adaptations of published NIDS and streaming baselines.

Every adapter runs under the DASP-v2 label contract. The labelled warm-up
windows come first. After that, one window late, an adapter receives the
window's features plus labels for the 128 label-blind uniform queries. The
interface never passes the other labels, so an adapter cannot read them.

Sources (pinned commits in revision/redesign_v2_20261018/official_sources):

* AOC-IDS, Zhang et al., INFOCOM 2024, xinchen930/AOC-IDS@41bfd751:
  autoencoder trained with the CRC contrastive loss. It pseudo-labels each
  new window with a two-Gaussian fit on cosine similarity to the normal
  prototype, then retrains on its history.
* CND-IDS, Fuhrman et al., DAC 2025, Sean-Fuhrman/CND-IDS@a09eecf2: a
  continual feature extractor trained with cluster pseudo-labels, a
  semi-hard triplet loss, reconstruction and LwF. Scores come from PCA
  reconstruction error fitted on normal data.
* SOUL, Amalapuram et al., 2024, amalapuram/SOUL@be0d4eb5: semi-supervised
  continual MLP with class-balanced replay, pseudo-label distillation on
  unlabelled data and gradient projection memory (GPM).
* ARF, Gomes et al., 2017, River ``forest.ARFClassifier``.

Adaptations shared by all four (reported as ``*-adapted``):

* No task boundaries are given. Each window is one task, labels arrive one
  window late and only the 128 queried samples are labelled.
* Warm-up windows are fully labelled, as for every other method.
* History buffers are bounded by ``dasp.memory_size`` (the replay budget of
  the other methods) instead of growing without limit.
* Decision thresholds that the sources pick on labelled test or validation
  data are picked on the queried labels already received.
* The three neural sources min-max scale their inputs. Here the scaler is
  fitted on the labelled warm-up windows only and clipped to [0, 1]: the
  prepared streams are z-scored with heavy tails, and the source networks
  have no normalisation layer. ARF splits on raw values and is unscaled.
"""
from __future__ import annotations

import copy
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import optimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ADAPTER_VERSION = "sota-adapters-20261019-r0"
ATTACK = 1


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Score threshold (attack when score >= t) maximising binary Macro-F1,
    the reported metric. None when the labels or scores cannot rank."""

    y = np.asarray(y) != 0
    if y.all() or not y.any() or np.ptp(scores) == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y[order]
    positives, negatives = int(y.sum()), int((~y).sum())
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(~sorted_y)
    fn, tn = positives - tp, negatives - fp
    attack_f1 = 2 * tp / (2 * tp + fp + fn)
    benign_f1 = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros(len(tn)), where=(2 * tn + fp + fn) > 0)
    # Only cut between distinct scores.
    last = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    macro = np.where(last, (attack_f1 + benign_f1) / 2, -1.0)
    return float(sorted_scores[int(np.argmax(macro))])


class ClassBalancedStore:
    """Bounded labelled store: equal reservoir share per class (ECBRS-like)."""

    def __init__(self, size: int, n_classes: int, seed: int):
        self.capacity = max(1, int(size) // n_classes)
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.x: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
        self.seen = [0] * n_classes

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        for row, label in zip(x, y.astype(int)):
            bucket = self.x[label]
            self.seen[label] += 1
            if len(bucket) < self.capacity:
                bucket.append(row)
            else:
                slot = int(self.rng.integers(self.seen[label]))
                if slot < self.capacity:
                    bucket[slot] = row

    def arrays(self, label: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        labels = range(self.n_classes) if label is None else [label]
        xs = [np.asarray(self.x[c]) for c in labels if self.x[c]]
        ys = [np.full(len(self.x[c]), c) for c in labels if self.x[c]]
        if not xs:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
        return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self.x)


class Reservoir:
    """Uniform bounded sample of everything seen (approximates full history)."""

    def __init__(self, size: int, seed: int):
        self.size = int(size)
        self.rng = np.random.default_rng(seed)
        self.items: list[tuple[np.ndarray, int]] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        for row, label in zip(x, y.astype(int)):
            self.seen += 1
            if len(self.items) < self.size:
                self.items.append((row, label))
            else:
                slot = int(self.rng.integers(self.seen))
                if slot < self.size:
                    self.items[slot] = (row, label)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.items:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
        return (np.stack([row for row, _ in self.items]).astype(np.float32),
                np.asarray([label for _, label in self.items], dtype=np.int64))


class Adapter:
    """Interface the runner drives. Labels arrive only through ``warmup``
    (labelled warm-up windows) and ``update`` (queried samples)."""

    method = "adapter"

    def __init__(self, cfg, n_features: int, n_classes: int, seed: int):
        if n_classes != 2:
            raise ValueError(f"{self.method} is a binary detector; got {n_classes} classes")
        self.cfg, self.n_features, self.seed = cfg, n_features, seed
        self.rng = np.random.default_rng(seed)
        self.recent: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=cfg.sota.calibration_windows)
        self._warmup_x: list[np.ndarray] = []
        self._warmup_y: list[np.ndarray] = []
        self.ready = False

    def warmup(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        self._warmup_x.append(_np(x).astype(np.float32))
        self._warmup_y.append(_np(y).astype(np.int64))
        return {}

    def end_warmup(self) -> dict:
        x, y = np.concatenate(self._warmup_x), np.concatenate(self._warmup_y)
        self._warmup_x, self._warmup_y = [], []
        self._fit_scaler(x)
        info = self._fit_warmup(self.transform(x), y)
        self.ready = True
        return info

    def predict(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        x = _np(x).astype(np.float32)
        if not self.ready:
            return np.zeros(len(x), dtype=np.int64), np.zeros(len(x))
        return self._predict(self.transform(x))

    def update(self, x: torch.Tensor, query: torch.Tensor, y_query: torch.Tensor) -> dict:
        x = self.transform(_np(x).astype(np.float32))
        query = _np(query).astype(np.int64)
        y_query = _np(y_query).astype(np.int64)
        self.recent.append((x[query], y_query))
        return self._update(x, query, y_query)

    def _fit_scaler(self, x: np.ndarray) -> None:
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def model_bytes(self) -> int:
        modules = [value for value in vars(self).values() if isinstance(value, nn.Module)]
        modules += [m for value in vars(self).values() if isinstance(value, list)
                    for m in value if isinstance(m, nn.Module)]
        return int(sum(p.numel() * p.element_size() for m in modules for p in m.parameters()))

    def memory_bytes(self) -> int:
        stores = [value for value in vars(self).values() if isinstance(value, (ClassBalancedStore, Reservoir))]
        return int(sum(len(store) for store in stores) * (self.n_features * 4 + 8))

    def recent_labels(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.concatenate([x for x, _ in self.recent]),
                np.concatenate([y for _, y in self.recent]))

    # Subclasses implement these.
    def _fit_warmup(self, x: np.ndarray, y: np.ndarray) -> dict:
        raise NotImplementedError

    def _predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _update(self, x: np.ndarray, query: np.ndarray, y_query: np.ndarray) -> dict:
        raise NotImplementedError


class TorchAdapter(Adapter):
    def __init__(self, cfg, n_features, n_classes, seed):
        super().__init__(cfg, n_features, n_classes, seed)
        self.generator = torch.Generator().manual_seed(seed)

    def _fit_scaler(self, x: np.ndarray) -> None:
        self.low = x.min(axis=0)
        self.span = np.where(x.max(axis=0) > self.low, x.max(axis=0) - self.low, 1.0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "low"):
            return x
        return np.clip((x - self.low) / self.span, 0.0, 1.0).astype(np.float32)

    def batches(self, n: int, batch_size: int):
        order = torch.randperm(n, generator=self.generator)
        for start in range(0, n, batch_size):
            yield order[start:start + batch_size]


# --------------------------------------------------------------------- AOC-IDS
class AOCAutoencoder(nn.Module):
    """Architecture of AOC-IDS utils.AE."""

    def __init__(self, input_dim: int):
        super().__init__()
        power = 2 ** round(math.log2(input_dim))
        half, quarter = power // 2, power // 4
        self.encoder = nn.Sequential(nn.Linear(input_dim, half), nn.ReLU(), nn.Linear(half, quarter))
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(quarter, half), nn.ReLU(), nn.Linear(half, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


def crc_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor | None:
    """AOC-IDS utils.CRCLoss. None when the batch holds no normal sample."""

    normal = labels == 0
    if int(normal.sum()) == 0:
        return None
    features = F.normalize(features, p=2, dim=1)
    logits = features @ features.T / temperature
    logits = logits * (1.0 - torch.eye(len(labels)))
    normal_rows = logits[normal]
    normal_normal = normal_rows[:, normal]
    normal_abnormal = normal_rows[:, ~normal]
    denominator = torch.exp(normal_normal) + torch.exp(normal_abnormal).sum()
    loss = -(normal_normal - torch.log(denominator)) * temperature
    return loss.mean()


def _two_gaussian_fit(values: np.ndarray, normal: np.ndarray, abnormal: np.ndarray):
    """AOC-IDS utils.evaluate: equal-weight two-Gaussian MLE (Nelder-Mead),
    initialised from the pseudo-labelled normal and abnormal similarities."""

    values, normal, abnormal = (np.asarray(v, dtype=np.float64) for v in (values, normal, abnormal))

    def init(part):
        part = part if len(part) >= 2 else values
        return float(np.mean(part)), float(np.std(part)) + 1e-3

    def nll(params):
        mu1, s1, mu2, s2 = params
        s1, s2 = abs(s1) + 1e-6, abs(s2) + 1e-6
        p1 = np.exp(-0.5 * ((values - mu1) / s1) ** 2) / (np.sqrt(2 * np.pi) * s1)
        p2 = np.exp(-0.5 * ((values - mu2) / s2) ** 2) / (np.sqrt(2 * np.pi) * s2)
        return -np.sum(np.log(0.5 * p1 + 0.5 * p2 + 1e-300))

    start = np.array([*init(normal), *init(abnormal)])
    mu1, s1, mu2, s2 = optimize.minimize(nll, start, method="Nelder-Mead").x
    s1, s2 = abs(s1) + 1e-6, abs(s2) + 1e-6
    # The component with the larger mean similarity to the normal prototype is normal.
    return ((mu1, s1), (mu2, s2)) if mu1 > mu2 else ((mu2, s2), (mu1, s1))


def _log_pdf(values, mu, sigma):
    return -0.5 * ((values - mu) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi) * sigma)


class AOCIDSAdapter(TorchAdapter):
    method = "aoc_ids_adapted"
    batch_size = 128

    def __init__(self, cfg, n_features, n_classes, seed):
        super().__init__(cfg, n_features, n_classes, seed)
        self.model = AOCAutoencoder(n_features)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.training.learning_rate)
        # History with the labels AOC trains on (true or flipped pseudo) and
        # the unflipped detection labels used to initialise the Gaussian fit.
        self.history = Reservoir(cfg.dasp.memory_size, seed + 11)
        self.labelled = ClassBalancedStore(cfg.dasp.memory_size, 2, seed + 13)

    def _train(self, x: np.ndarray, y: np.ndarray, epochs: int) -> int:
        tx, ty = torch.from_numpy(x), torch.from_numpy(y)
        self.model.train()
        processed = 0
        for _ in range(epochs):
            for index in self.batches(len(tx), self.batch_size):
                features, recon = self.model(tx[index])
                parts = [crc_loss(features, ty[index], self.cfg.sota.aoc_temperature),
                         crc_loss(recon, ty[index], self.cfg.sota.aoc_temperature)]
                if parts[0] is None:
                    continue
                self.optimizer.zero_grad()
                (parts[0] + parts[1]).backward()
                self.optimizer.step()
                processed += len(index)
        return processed

    @torch.no_grad()
    def _detect(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        normals, _ = self.labelled.arrays(0)
        ref_x, ref_y = self.history.arrays()
        if not len(normals) or not len(ref_x):
            return np.zeros(len(x), dtype=np.int64), np.zeros(len(x))
        tn, tr, tx = map(torch.from_numpy, (normals, ref_x, x))
        votes, margins = [], []
        for output in (0, 1):  # encoder features, then reconstruction
            prototype = F.normalize(self.model(tn)[output], dim=1).mean(0, keepdim=True)
            ref = F.cosine_similarity(F.normalize(self.model(tr)[output], dim=1), prototype).numpy()
            test = F.cosine_similarity(F.normalize(self.model(tx)[output], dim=1), prototype).numpy()
            (mu_n, s_n), (mu_a, s_a) = _two_gaussian_fit(ref, ref[ref_y == 0], ref[ref_y == 1])
            pdf_n, pdf_a = np.exp(_log_pdf(test, mu_n, s_n)), np.exp(_log_pdf(test, mu_a, s_a))
            votes.append((pdf_a > pdf_n).astype(np.int64))
            margins.append(np.abs(pdf_a - pdf_n))
        # AOC: take the head (encoder or decoder) that is more certain.
        prediction = np.where(margins[0] > margins[1], votes[0], votes[1])
        return prediction, prediction.astype(float)

    def _fit_warmup(self, x, y):
        processed = self._train(x, y, self.cfg.sota.warmup_epochs)
        self.history.add(x, y)
        self.labelled.add(x, y)
        return {"gradient_evaluations": processed}

    def _predict(self, x):
        return self._detect(x)

    def _update(self, x, query, y_query):
        pseudo, _ = self._detect(x)
        labels = pseudo.copy()
        unlabelled = np.setdiff1d(np.arange(len(x)), query)
        flips = self.rng.choice(unlabelled, int(self.cfg.sota.aoc_flip_fraction * len(unlabelled)),
                                replace=False)
        labels[flips] = 1 - labels[flips]
        labels[query] = y_query
        history_x, history_y = self.history.arrays()
        train_x = np.concatenate([history_x, x]) if len(history_x) else x
        train_y = np.concatenate([history_y, labels]) if len(history_y) else labels
        processed = self._train(train_x, train_y, self.cfg.sota.epochs)
        detection = pseudo.copy()
        detection[query] = y_query
        self.history.add(x, detection)
        self.labelled.add(x[query], y_query)
        return {"gradient_evaluations": processed, "sample_updates": len(train_x),
                "pseudo_labels_used": len(unlabelled)}


# --------------------------------------------------------------------- CND-IDS
def knee_cluster_count(options: list[int], inertia: list[float]) -> int:
    """Elbow of a convex decreasing curve: the point farthest below the chord
    (Kneedle's difference curve). Falls back to the largest option."""

    x = np.asarray(options, dtype=float)
    y = np.asarray(inertia, dtype=float)
    if len(x) < 3 or np.ptp(y) == 0:
        return int(options[-1])
    xn, yn = (x - x[0]) / np.ptp(x), (y - y.min()) / np.ptp(y)
    chord = yn[0] + (yn[-1] - yn[0]) * xn
    gap = chord - yn
    return int(options[int(np.argmax(gap))]) if gap.max() > 0 else int(options[-1])


def semihard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """pytorch-metric-learning TripletMarginLoss + semi-hard miner with
    LpDistance(normalize_embeddings=True) and ThresholdReducer(low=0)."""

    distance = torch.cdist(F.normalize(embeddings, dim=1), F.normalize(embeddings, dim=1))
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool)
    d_ap, d_an = distance[:, :, None], distance[:, None, :]
    valid = positive[:, :, None] & ~same[:, None, :]
    valid = valid & (d_an > d_ap) & (d_an < d_ap + margin)
    losses = F.relu(d_ap - d_an + margin)[valid]
    losses = losses[losses > 0]
    return losses.mean() if len(losses) else embeddings.sum() * 0.0


class CNDIDSNet(nn.Module):
    """Architecture of CND-IDS FeatureExtractors/CND_IDS.py."""

    def __init__(self, n_input: int, n_latent: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_input, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_latent))
        self.decoder = nn.Sequential(nn.Linear(n_latent, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_input))


class CNDIDSAdapter(TorchAdapter):
    method = "cnd_ids_adapted"
    batch_size = 64

    def __init__(self, cfg, n_features, n_classes, seed):
        super().__init__(cfg, n_features, n_classes, seed)
        self.net = CNDIDSNet(n_features, cfg.sota.cnd_latent_dim)
        self.teachers: list[nn.Module] = []
        # The normal pool CND-IDS fits its scorer on: benign labels it was given.
        self.normals = ClassBalancedStore(cfg.dasp.memory_size, 1, seed + 17)
        self.pca: PCA | None = None
        self.threshold = np.inf

    @torch.no_grad()
    def encode(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        return self.net.encoder(torch.from_numpy(x)).numpy()

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sota = self.cfg.sota
        embeddings = self.net.encoder(x)
        loss = semihard_triplet_loss(embeddings, y, sota.cnd_margin)
        for teacher in self.teachers:
            with torch.no_grad():
                target = teacher(x)
            loss = loss + sota.cnd_lwf_strength * sota.cnd_reg_strength * F.mse_loss(embeddings, target)
        return loss + sota.cnd_reg_strength * F.mse_loss(self.net.decoder(embeddings), x)

    def _fit(self, x: np.ndarray, y: np.ndarray, epochs: int) -> int:
        """CND_IDS.fit: first fifth held out, best validation epoch kept."""

        order = self.rng.permutation(len(x))
        x, y = torch.from_numpy(x[order]), torch.from_numpy(y[order])
        split = len(x) // 5
        val_x, val_y, tr_x, tr_y = x[:split], y[:split], x[split:], y[split:]
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.training.learning_rate)
        best, best_state, processed = np.inf, copy.deepcopy(self.net.state_dict()), 0
        for _ in range(epochs):
            self.net.train()
            for index in self.batches(len(tr_x), self.batch_size):
                optimizer.zero_grad()
                self._loss(tr_x[index], tr_y[index]).backward()
                optimizer.step()
                processed += len(index)
            self.net.eval()
            with torch.no_grad():
                losses = [float(self._loss(val_x[i:i + self.batch_size], val_y[i:i + self.batch_size]))
                          for i in range(0, len(val_x), self.batch_size)]
            if losses and float(np.mean(losses)) < best:
                best, best_state = float(np.mean(losses)), copy.deepcopy(self.net.state_dict())
        self.net.load_state_dict(best_state)
        self.teachers.append(copy.deepcopy(self.net.encoder).eval())
        self.teachers = self.teachers[-self.cfg.sota.cnd_max_teachers:]
        return processed

    def _fit_scorer(self) -> None:
        """PCA novelty scorer on the encoded normal pool (90/10 split as in
        CND-IDS datastream.init_normal / init_val; 95% explained variance)."""

        normals, _ = self.normals.arrays(0)
        order = self.rng.permutation(len(normals))
        cut = max(1, int(0.9 * len(normals)))
        fit, held = normals[order[:cut]], normals[order[cut:]]
        encoded = self.encode(fit)
        explained = np.cumsum(PCA().fit(encoded).explained_variance_ratio_)
        self.pca = PCA(n_components=int(np.argmax(explained >= 0.95)) + 1, svd_solver="full").fit(encoded)
        if self.cfg.sota.cnd_threshold == "quantile":
            reference = self.score(held if len(held) else fit)
            self.threshold = float(np.quantile(reference, self.cfg.sota.cnd_quantile))
        elif self.cfg.sota.cnd_threshold == "labelled_f1":
            if self.recent:
                x, y = self.recent_labels()
                threshold = best_f1_threshold(self.score(x), y)
            else:
                threshold = None
            if threshold is None:  # warm-up: use the labelled warm-up data instead
                x, y = self._calibration
                threshold = best_f1_threshold(self.score(x), y)
            self.threshold = np.inf if threshold is None else threshold
        else:
            raise ValueError(f"unknown CND threshold {self.cfg.sota.cnd_threshold!r}")

    def score(self, x: np.ndarray) -> np.ndarray:
        encoded = self.encode(x)
        reconstructed = self.pca.inverse_transform(self.pca.transform(encoded))
        return np.abs(encoded - reconstructed).mean(axis=1)

    def _cluster_labels(self, x: np.ndarray) -> np.ndarray:
        """CND-IDS K_Means labeller: clusters holding a known normal are normal.

        Cluster counts are capped at an eighth of the window, and centroids
        start at random samples: k-means++ seeding was ~90% of the cost."""

        cap = max(2, len(x) // 8)
        options = sorted({min(k, cap) for k in self.cfg.sota.cnd_cluster_options})
        fits = [KMeans(n_clusters=k, init="random", n_init=1, max_iter=100, random_state=self.seed).fit(x)
                for k in options]
        chosen = fits[options.index(knee_cluster_count(options, [f.inertia_ for f in fits]))]
        normals, _ = self.normals.arrays(0)
        normal_clusters = set(chosen.predict(normals).tolist())
        return np.asarray([0 if c in normal_clusters else 1 for c in chosen.labels_], dtype=np.int64)

    def _fit_warmup(self, x, y):
        self.normals.add(x[y == 0], y[y == 0])
        self._calibration = (x, y)
        processed = self._fit(x, y, self.cfg.sota.warmup_epochs)
        self._fit_scorer()
        return {"gradient_evaluations": processed}

    def _predict(self, x):
        scores = self.score(x)
        return (scores >= self.threshold).astype(np.int64), scores

    def _update(self, x, query, y_query):
        benign = query[y_query == 0]
        self.normals.add(x[benign], np.zeros(len(benign), dtype=np.int64))
        labels = self._cluster_labels(x)
        pseudo = len(x) - len(query)
        labels[query] = y_query
        processed = self._fit(x, labels, self.cfg.sota.epochs)
        self._fit_scorer()
        return {"gradient_evaluations": processed, "sample_updates": len(x), "pseudo_labels_used": pseudo}


# ------------------------------------------------------------------------ SOUL
class GPMNet(nn.Module):
    """Bias-free MLP that records each linear layer's input for GPM."""

    def __init__(self, n_input: int, hidden: list[int], n_classes: int):
        super().__init__()
        dims = [n_input, *hidden, n_classes]
        self.layers = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.inputs: list[torch.Tensor] = []

    def forward(self, x):
        self.inputs = []
        for index, layer in enumerate(self.layers):
            self.inputs.append(x.detach())
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.relu(x)
        return x


def update_gpm(bases: list[np.ndarray], activations: list[np.ndarray], thresholds: list[float]) -> list[np.ndarray]:
    """SOUL update_GPM (Saha et al. GPM, Eq. 5 and 8-9)."""

    result = []
    for index, activation in enumerate(activations):
        threshold = thresholds[index]
        if not bases:
            u, s, _ = np.linalg.svd(activation, full_matrices=False)
            ratio = np.cumsum(s ** 2) / (s ** 2).sum()
            result.append(u[:, :int(np.sum(ratio < threshold))])
            continue
        basis = bases[index]
        total = (np.linalg.svd(activation, compute_uv=False) ** 2).sum()
        residual = activation - basis @ (basis.T @ activation)
        u, s, _ = np.linalg.svd(residual, full_matrices=False)
        accumulated = (total - (s ** 2).sum()) / total
        rank = 0
        for value in (s ** 2) / total:
            if accumulated >= threshold:
                break
            accumulated += value
            rank += 1
        merged = np.hstack([basis, u[:, :rank]]) if rank else basis
        result.append(merged[:, :merged.shape[0]])
    return result


class SOULAdapter(TorchAdapter):
    method = "soul_adapted"
    batch_size = 128

    def __init__(self, cfg, n_features, n_classes, seed):
        super().__init__(cfg, n_features, n_classes, seed)
        self.model = GPMNet(n_features, list(cfg.model.hidden_dims), 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.training.learning_rate, momentum=0.9,
                                         nesterov=True, weight_decay=cfg.training.weight_decay)
        self.memory = ClassBalancedStore(cfg.dasp.memory_size, 2, seed + 19)
        self.bases: list[np.ndarray] = []
        self.threshold = 0.5
        n_layers = len(self.model.layers)
        self.gpm_thresholds = [cfg.sota.soul_gpm_threshold] + [cfg.sota.soul_gpm_hidden_threshold] * (n_layers - 1)

    def _memory_batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """sample_batch_from_memory: a fixed share of the batch is attack."""

        if size <= 0 or not len(self.memory):
            return None
        parts = []
        attack_size = max(1, math.floor(size * self.cfg.sota.soul_minority_allocation))
        for label, count in ((ATTACK, attack_size), (0, max(1, size - attack_size))):
            x, y = self.memory.arrays(label)
            if len(x):
                pick = self.rng.choice(len(x), min(count, len(x)), replace=False)
                parts.append((x[pick], y[pick]))
        if not parts:
            return None
        return (torch.from_numpy(np.concatenate([p[0] for p in parts])),
                torch.from_numpy(np.concatenate([p[1] for p in parts])))

    def _project(self) -> None:
        for layer, basis in zip(self.model.layers, self.bases):
            if basis.shape[1]:
                projector = torch.from_numpy(basis @ basis.T).float()
                layer.weight.grad -= layer.weight.grad @ projector

    def _train(self, lab_x, lab_y, unl_x, epochs, distill: bool) -> int:
        """SOUL train(): labelled + replay + unlabelled sub-batches per step."""

        memory_size = math.floor(self.batch_size * self.cfg.sota.soul_memory_batch_fraction)
        rest = self.batch_size - memory_size
        if distill and len(unl_x):
            ratio = len(lab_x) / (len(lab_x) + len(unl_x))
            lab_size = max(1, math.floor(rest * ratio))
            unl_size = rest - lab_size
            steps = max(math.ceil(len(lab_x) / lab_size), math.ceil(len(unl_x) / unl_size))
        else:
            lab_size, unl_size = rest, 0
            steps = math.ceil(len(lab_x) / lab_size)
        lab_x, lab_y = torch.from_numpy(lab_x), torch.from_numpy(lab_y)
        unl = torch.from_numpy(unl_x) if unl_size else None
        processed = 0
        self.model.train()
        for _ in range(epochs):
            lab_order = torch.randperm(len(lab_x), generator=self.generator)
            unl_order = torch.randperm(len(unl), generator=self.generator) if unl is not None else None
            for step in range(steps):
                pick = lab_order[(torch.arange(lab_size) + step * lab_size) % len(lab_x)]
                x, y = lab_x[pick], lab_y[pick]
                replay = self._memory_batch(memory_size)
                if replay is not None:
                    x, y = torch.cat([x, replay[0]]), torch.cat([y, replay[1]])
                loss = F.binary_cross_entropy(F.softmax(self.model(x), dim=1), F.one_hot(y, 2).float())
                processed += len(x)
                if unl is not None:
                    batch = unl[unl_order[(torch.arange(unl_size) + step * unl_size) % len(unl)]]
                    probability = F.softmax(self.model(batch), dim=1)
                    target = F.one_hot(probability.detach().argmax(1), 2).float()
                    loss = loss + self.cfg.sota.soul_alpha * F.binary_cross_entropy(probability, target)
                    processed += len(batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.bases:
                    self._project()
                self.optimizer.step()
        return processed

    @torch.no_grad()
    def _attack_probability(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        return F.softmax(self.model(torch.from_numpy(x)), dim=1)[:, ATTACK].numpy()

    @torch.no_grad()
    def _grow_gpm(self, x: np.ndarray, y: np.ndarray) -> None:
        """get_representation_matrix: activations of labelled attack samples."""

        attack = x[y == ATTACK]
        if len(attack) < 2:
            return
        pick = self.rng.choice(len(attack), min(len(attack), self.cfg.sota.soul_gpm_samples), replace=False)
        self.model.eval()
        self.model(torch.from_numpy(attack[pick]))
        activations = [value.numpy().T.astype(np.float64) for value in self.model.inputs]
        self.bases = update_gpm(self.bases, activations, self.gpm_thresholds)

    def _calibrate(self, x: np.ndarray, y: np.ndarray) -> None:
        """SOUL picks its operating point on labelled validation data."""

        threshold = best_f1_threshold(self._attack_probability(x), y)
        if threshold is not None:
            self.threshold = threshold

    def _fit_warmup(self, x, y):
        processed = self._train(x, y, np.empty((0, x.shape[1]), dtype=np.float32),
                                self.cfg.sota.warmup_epochs, distill=False)
        self.memory.add(x, y)
        self._grow_gpm(x, y)
        self._calibrate(x, y)
        return {"gradient_evaluations": processed}

    def _predict(self, x):
        probability = self._attack_probability(x)
        return (probability >= self.threshold).astype(np.int64), probability

    def _update(self, x, query, y_query):
        unlabelled = np.setdiff1d(np.arange(len(x)), query)
        processed = self._train(x[query], y_query, x[unlabelled], self.cfg.sota.epochs, distill=True)
        self.memory.add(x[query], y_query)
        self._grow_gpm(x[query], y_query)
        self._calibrate(*self.recent_labels())
        protected = [int(b.shape[1]) for b in self.bases]
        return {"gradient_evaluations": processed, "sample_updates": processed,
                "pseudo_labels_used": len(unlabelled),
                "gpm_protected_fraction": float(np.mean([p / layer.in_features for p, layer
                                                         in zip(protected, self.model.layers)])) if protected else 0.0}


# ------------------------------------------------------------------------- ARF
class ARFAdapter(Adapter):
    method = "arf_adapted"

    def __init__(self, cfg, n_features, n_classes, seed):
        super().__init__(cfg, n_features, n_classes, seed)
        try:
            from river import forest
        except ImportError as error:  # pragma: no cover - dependency message
            raise RuntimeError("arf_adapted needs River: pip install river==0.23.0") from error
        max_features = cfg.sota.arf_max_features
        if max_features not in ("sqrt", "log2"):
            max_features = float(max_features)
        self.model = forest.ARFClassifier(n_models=cfg.sota.arf_n_models, max_features=max_features,
                                          lambda_value=cfg.sota.arf_lambda, grace_period=cfg.sota.arf_grace_period,
                                          leaf_prediction=cfg.sota.arf_leaf_prediction, seed=seed)
        self.ready = True  # an online learner predicts from the first sample

    @staticmethod
    def _records(x: np.ndarray) -> list[dict[int, float]]:
        return [dict(enumerate(row.tolist())) for row in x]

    def _learn(self, x: np.ndarray, y: np.ndarray) -> int:
        for record, label in zip(self._records(x), y.tolist()):
            self.model.learn_one(record, int(label))
        return len(y)

    def warmup(self, x, y):
        # ARF learns every labelled warm-up sample as it arrives, like the MLPs.
        return {"sample_updates": self._learn(_np(x).astype(np.float32), _np(y).astype(np.int64))}

    def end_warmup(self):
        return {}

    def _predict(self, x):
        probabilities = [self.model.predict_proba_one(record) for record in self._records(x)]
        scores = np.asarray([p.get(ATTACK, 0.0) for p in probabilities])
        # predict_one is the arg-max of predict_proba_one; benign when empty.
        predictions = np.asarray([int(p.get(ATTACK, 0.0) > p.get(0, 0.0)) for p in probabilities])
        return predictions.astype(np.int64), scores

    def model_bytes(self) -> int:
        import pickle
        return len(pickle.dumps(self.model, protocol=pickle.HIGHEST_PROTOCOL))

    def _counter(self, name: str) -> int:
        value = getattr(self.model, name, 0)
        return int(value() if callable(value) else value)

    def _update(self, x, query, y_query):
        return {"sample_updates": self._learn(x[query], y_query),
                "internal_drifts": self._counter("n_drifts_detected"),
                "internal_warnings": self._counter("n_warnings_detected")}


ADAPTERS = {cls.method: cls for cls in (AOCIDSAdapter, CNDIDSAdapter, SOULAdapter, ARFAdapter)}


def build(cfg, n_features: int, n_classes: int, seed: int) -> Adapter:
    return ADAPTERS[cfg.dasp.baseline](cfg, n_features, n_classes, seed)
