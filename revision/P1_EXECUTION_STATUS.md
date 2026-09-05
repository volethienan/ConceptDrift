# Phase F P1 Execution Status

Status date: September 2, 2026.

## Locked Scope

- Component ablations: B1--B5 across 5 datasets, 3 scenarios, and 3 stream seeds; 225 runs.
- Feature-order robustness: A1, A6, and A9 under permutation seeds 3101--3103; 405 runs.
- Natural chronological evaluation: A1, A10, experience replay, and online MLP under model seeds 91--93; 12 runs.
- Total full P1 runs: 642.
- Pre-run snapshot: `revision/full_p1_lock_2026-09-02`.
- Snapshot-manifest SHA-256: `8011931FD8041AC374042ED3CBEE55D5C3004E4DB25417794B38525CDD3123A5`.

## Pre-run Gates

- Focused revision tests: 23/23 passed before the full P1 run.
- Component smoke: 5/5 runs and 1/1 group passed.
- Feature-permutation smoke: 3/3 runs and 1/1 group passed.
- Natural chronological smoke: 4/4 runs and 1/1 group passed.
- The full legacy test suite remains blocked at collection by four pre-existing tests that import three absent legacy tool modules; the P1-focused suite is unaffected.

## Natural Stream Construction

- Raw source: `data/NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv`.
- Raw size: 5,302,886,266 bytes.
- Raw SHA-256: `53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78`.
- Construction: stable global timestamp ordering followed by the earliest 147,456 valid rows.
- Layout: 36 windows of 4,096 rows; first 4 windows used for warmup only.
- Prepared-stream SHA-256: `ea970e87e6a9da60a51d9d0f2bf16df9825396b795d9d8ad6d1321fd8abee6085`.
- Preparation runtime: 124.0 seconds.

## Full Execution

| Family | Runs | Validated groups | Failed groups | Runtime (s) | Validation SHA-256 | Artifact-manifest SHA-256 |
|---|---:|---:|---:|---:|---|---|
| Components | 225/225 | 45/45 | 0 | 628.0 | `939C8364A6C1C877D125D91CD7EAAB41696A4261D3AE063ECC186B4D60B76609` | `20DEAB9ED0CFF0F4BFCE552A242122C913EDB59665B2AD31794F6B5E5F9587AB` |
| Feature permutation | 405/405 | 135/135 | 0 | 1,623.4 | `941650B9033265811F142A4872B17F2D963099E5B05E13816B42BBA6CDE51DC6` | `E08531EC21882ECDC22982C28E4C40046B28F9D2A99E756069991E0371ED188D` |
| Natural chronological | 12/12 | 3/3 | 0 | 196.4 | `CD3E618A8528E1106D8C6FA363B1B747DCEF9F53157D97FB3648A36428AADCFA` | `0A80586A247BB642DCCBFAD2165C384B9B5A37B0ED2A2CD6C1E2C4C6DBAA38BF` |

Total full P1 runtime was 2,447.8 seconds, approximately 40 minutes 48 seconds, excluding smoke tests and natural-stream preparation.

## Cluster-Aware Analysis

- P0 inference unit: 50 `dataset × scenario` clusters after averaging the 10 seed/model cells within each cluster.
- P1 component and feature-order inference unit: 15 `dataset × scenario` clusters.
- Confidence intervals: 20,000 cluster-bootstrap replicates.
- Multiplicity: Holm correction applied separately to sign-test and Wilcoxon families.
- Leave-one-dataset-out results are reported for all P0 contrasts.
- Analysis artifact-manifest SHA-256: `2D0059899FE5C77B98DB72971F265E3D5D5F7664FB484C54287AA0BF401ABB25`.
- Full report: `outputs_revision_v1/cluster_aware_v1/REPORT.md`.

## Main Findings for Revision Claims

1. Removing replay reduced Macro-F1 by 0.03135 on average across the 15 P1 clusters; the 95% cluster-bootstrap interval was [-0.05204, -0.01421]. The Holm-adjusted Wilcoxon p-value was 0.0410.
2. Removing the performance gate reduced Macro-F1 by 0.05159; the 95% interval was [-0.07790, -0.02659]. Both cluster sign and Wilcoxon results remained significant after Holm correction.
3. Setting new-sample weighting to one and replacing class-balanced selection with uniform label-blind selection did not produce a detectable cluster-level change under this P1 subset.
4. The parameter-matched flat global model B5 did not differ detectably from modular all-module adaptation A6; the mean difference was -0.00033 with a 95% interval of [-0.00464, 0.00450].
5. None of A1, A6, or A9 showed a detectable cluster-level loss under the three fixed feature permutations. This supports robustness to the tested feature orders but does not establish semantic coherence of the four subspaces.
6. On the natural chronological stream, mean Macro-F1 was 0.84253 for A1, 0.84953 for A10/SSF, 0.80911 for experience replay, and 0.82018 for online MLP. These are descriptive results from one operational stream, not independent-environment inference.
7. A1 required 2.95 seconds of mean total adaptation time on the natural stream versus 33.47 seconds for SSF, but its prediction-decision throughput was about 54,132 flows/s versus 167,708 for experience replay and 172,484 for online MLP. The defensible conclusion is therefore an accuracy/adaptation-cost/inference-throughput trade-off, not uniformly greater efficiency.
8. In the 50-cluster P0 analysis, A6 all-module adaptation exceeded A1 by 0.00098 Macro-F1 on average, and this contrast remained significant after Holm correction. The revised manuscript must not claim that attribution-guided routing is predictively superior to all-module routing under the locked protocol.

## Remaining Non-Experimental Work

- Revise claims, abstract, discussion, and response letter to match the cluster-aware and natural-stream results.
- Add exact archival repository/access details after the archive is published.
- Replace author-contribution placeholders and finalize funding and conflict-of-interest declarations using author-verified information.
- Move repeated implementation and audit detail to appendices or supplementary material.
