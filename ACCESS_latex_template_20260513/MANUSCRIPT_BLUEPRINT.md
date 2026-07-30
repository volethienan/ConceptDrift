# Current DASP manuscript blueprint

## Title and method

**DASP: Drift-Triggered Selective Layer Adaptation for Streaming Network
Intrusion Detection**

After one warm-up, DASP freezes the base 256--128--64 MLP. Online changes are
restricted to low-rank residual adapters, feature-wise residual modulation, an
input adapter, and the classifier. A stable precheck, rank-4 fast path, rank-32
strong path, current-batch similarity replay, class-balanced guard, and fused
inference cache form the current mechanism.

## Research questions

- **RQ1:** Does DASP improve early post-drift Macro-F1 and Attack Recall relative
  to exactly seven locked comparators?
- **RQ2:** How do drift type, recurrence, stationary activity, and detector
  matching explain the result?
- **RQ3:** What forward, complete test-then-train, throughput, update, and GPU
  memory costs accompany DASP?

## Evidence boundary

- Confirmation: NF-UNSW-NB15-v3, binary Benign/Attack, S0--S4, five stream
  seeds, two model seeds, eight methods, 400/400 completed runs.
- Systems: the same scenarios and methods, one stream seed, two model seeds,
  80/80 completed GPU runs.
- DASP mean EarlyMacroF1@5: 0.995543, highest observed.
- DASP versus Replay: +0.000477, 95% CI [-0.000060, 0.001019]; universal
  superiority is not established.
- DASP EarlyAttackRecall@5: 0.990353; SSF-faithful is higher by 0.004887 with a
  supported interval.
- Detector precision/recall: 0.1667/0.1800.
- P95 forward inference: 3.675 ms/window, rank 1/8.
- Mean complete latency: 47.558 ms/window, rank 5/8; throughput 87,041 flows/s.
- Base/global update violations: zero.

## Manuscript logic

1. Temporal leakage, delayed-label, and label-budget problem.
2. Related work on drift monitoring, replay, SSF, and RepShield.
3. Frozen-backbone layer-adapter mechanism and fused inference.
4. Locked NF-UNSW controlled protocol and exactly seven comparators.
5. Drift adaptation, paired intervals, recurrence, detector behavior, and S0.
6. Separate CUDA-synchronized systems results.
7. Discussion of accuracy versus detector validity and forward speed versus
   update tail.
8. Limitations, reproducibility, data availability, and disclosure statements.

## Figures

- Figure 1: prediction-first DASP pseudocode.
- Figure 2: early Macro-F1 heatmap by drift episode.
- Figure 3: confirmation accuracy versus complete GPU latency.

## Required next evidence

Run the same current DASP implementation on NF-CSE-CIC-IDS2018-v3,
NF-ToN-IoT-v3, NF-BoT-IoT-v3, and LUFlow, followed by natural chronological
streams. Do not pool outputs produced by earlier DASP implementations with the
current confirmation.
