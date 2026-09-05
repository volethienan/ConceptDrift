# Response to the Editor and Reviewers

Dear Dr. Prakasam Periasamy and Reviewers,

Thank you for the careful, constructive, and technically detailed evaluation of our manuscript, “DASP: Drift-Attributed Subspace Plasticity for Localized Continual Adaptation in Streaming Network Intrusion Detection.” The comments identified an important distinction that was not sufficiently clear in the original submission: evidence for the complete DASP system is not the same as evidence that its attribution router independently improves predictive performance.

We have revised the manuscript around that distinction. We preserve the original 1,200-run pilot and 4,000-run eight-method confirmation because those experiments remain valid under their documented method-specific supervision semantics. We now label their comparative conclusions explicitly as protocol-specific. We then add two independent reviewer-requested evidence layers: a 5,000-run equal-label-exposure routing/locality matrix and 642 supplementary component, feature-order, and natural-chronological runs. The new results lead to a narrower but better-supported claim: DASP is an auditable, representation-subspace-selective adaptation framework; replay and delayed-label performance gating are its clearest supported performance contributors; the present experiments do not demonstrate predictive superiority of combined attribution routing over all-module adaptation.

The detailed responses follow. Section and table names refer to the revised manuscript.

---

## Response to Reviewer 1

### Comment R1-1: Add routing ablations under the same repair and replay budgets

**Response:** We agree. This was the most important missing control in the original manuscript. We added a 5,000-run matrix containing combined DASP, deterministic random top-3, fixed-module top-3, MMD-only top-3, conflict-only top-3, all-module adaptation, k=1, k=2, classifier-only adaptation, and SSF. Conditions A1–A9 use the same modular initialization, queried-label indices, repair-candidate and gate pools, 128-example optimization subset, 512-example replay subset, trigger/gate rules, optimizer, learning rate, batch size, epoch count, classifier plasticity, and reference refresh. Only the representation routing policy or pre-registered k differs. SSF retains its own adaptation logic but uses the same total-label-access contract.

The result is informative and not uniformly favorable to the original routing interpretation. Combined routing is not significantly different from random, fixed, MMD-only, or conflict-only routing after Holm correction. All-module adaptation is higher than combined routing by 0.000977 Macro-F1 on average across 50 dataset–scenario clusters, with a 95% cluster-bootstrap interval of 0.000505 to 0.001460 and a Holm-adjusted Wilcoxon p value of 0.00199. We therefore removed the claim that the combined router is the demonstrated source of predictive superiority. The revised paper describes routing as a deterministic intervention-control and audit mechanism.

**Changes made:** Methods and Fairness Controls; Pilot and Confirmation Matrices; Equal-Exposure Routing and Locality Controls; RQ2 discussion; Threats to Validity; Conclusion; Table “Equal-label-exposure confirmation.”

### Comment R1-2: Address unequal total label exposure

**Response:** We agree that matching the number of optimization examples is not equivalent to matching total supervision. We did not retroactively relabel the original 4,000-run matrix as equal-label. Instead, we preserve it as a comparison of complete method realizations under documented method-specific supervision paths and explicitly qualify its claims.

For the new confirmation, every A1–A10 condition receives exactly the same 512 queried labels in each delayed window. The same query-index digest is verified across paired conditions. The labels are divided into a 384-example repair-candidate pool and a disjoint 128-example performance-gate pool; no more than 128 new examples enter optimization. This experiment directly addresses label-access fairness for DASP, its routing/locality controls, and the primary SSF comparator.

Under the equal-exposure contract, the DASP–SSF contrast is heterogeneous: the mean cluster difference is +0.001555, the median is −0.005160, and DASP wins 12 and loses 38 of 50 dataset–scenario clusters. We therefore make no robust equal-exposure superiority claim over SSF. This narrower statement coexists logically with the original result: the original experiment supports integrated, protocol-specific comparative effectiveness, whereas the new experiment addresses supervision fairness.

**Changes made:** Abstract; Problem Formulation; Methods and Fairness Controls; Equal-Exposure Routing and Locality Controls; Discussion; Limitations; Data/reproduction appendix.

### Comment R1-3: Separate routing, modularization, replay, new-sample weighting, and performance gating

**Response:** We added a pre-specified P1 component study with 225 runs over 15 dataset–scenario clusters. Removing replay reduces Macro-F1 by 0.031353 on average, with a 95% cluster-bootstrap interval of −0.052036 to −0.014208. Removing the performance gate reduces Macro-F1 by 0.051586, with an interval of −0.077903 to −0.026589. Both effects remain significant in the Holm-adjusted Wilcoxon family. These results identify replay and performance gating as the strongest supported components.

Setting the new-sample weight to one produces a mean difference of +0.001233 with an interval spanning zero. Uniform label-blind repair selection produces −0.001167 with an interval spanning zero. We therefore do not claim that either tested choice independently improves performance in this subset. A parameter-matched flat global model differs from modular all-module adaptation by −0.000328, also with an interval spanning zero. Because this control changes both the representation layout and the representation used for MMD, we interpret it narrowly rather than claiming architectural superiority.

**Changes made:** Component Isolation subsection and table; RQ2 discussion; Abstract; Conclusion; revised mechanistic claims throughout.

### Comment R1-4: Strengthen real-world validity or moderate real-world claims

**Response:** We did both. We added an unmodified chronological NF-ToN-IoT-v3 evaluation constructed by stable global timestamp ordering, without injected scenario transformations. The earliest 147,456 valid rows form 36 windows, of which four are used for warm-up. Results are averaged over three model seeds and are described as descriptive because only one operational stream is available.

DASP obtains Macro-F1 0.842529, compared with 0.849527 for SSF, 0.809106 for Experience Replay, and 0.820182 for Online MLP. DASP requires 2.945 seconds of adaptation versus 33.472 seconds for SSF, but its inference throughput is about 54,132 flows/s compared with 167,708–172,484 flows/s for the flat MLP baselines. We now present this as an accuracy/adaptation-cost/inference-throughput trade-off. We also state explicitly that the controlled transformations are stress tests and not proof that every episode reproduces operational concept drift.

**Changes made:** Unmodified Natural Chronological Stream subsection and table; Abstract; RQ4 discussion; Limitations; Conclusion.

### Comment R1-5: Improve reproducibility and provide actual repository/environment details

**Response:** We expanded the reproducibility package and corrected the repository address to the verified Git remote. The public base repository and original confirmation are available at https://github.com/volethienan/ConceptDrift, commit 42d9cf3cbce4133ba4b1f13e3a787f46b9182cd9. The original 4,000-run evidence is protected by a 12,518-file SHA-256 manifest and a run-to-artifact map. Separate local P0 and P1 locks contain source snapshots, configurations, environment captures, validation records, and output manifests. Because these revision files are newer than the public base commit, the hash-locked revision bundle must be uploaded as a tagged release or attached to the resubmission before this response is submitted; the final release identifier is intentionally left as an author-completion field rather than being fabricated.

The revised manuscript and README give executable commands for the original pilot/confirmation, the 5,000-run P0 matrix, the three P1 families, cluster-aware analysis, and verification of the original evidence lock. The recorded environment is CPython 3.12.7, PyTorch 2.3.1+cpu, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, and SciPy 1.16.2 on Windows 11 build 26200, using an Intel Core i5-13420H CPU, 8 physical/12 logical cores, approximately 16 GB RAM, CPU-only execution, and one PyTorch thread per run. Exact package versions are pinned in requirements-revision-lock.txt.

**Changes made:** Reproducibility and Performance Measurement; Data Availability; Reproduction and Archival Guide; local reproduction guide; archived lock and manifest files. Final tagged-release upload remains an author action before resubmission.

### Comment R1-6: Complete author contributions, funding, and conflict-of-interest metadata

**Response:** We agree that placeholders cannot remain in the resubmitted manuscript. These declarations concern author-specific facts that cannot be inferred safely from code or experiment records. The corresponding sections have been isolated for final author verification. Before resubmission, all three authors must approve the CRediT roles, funding statement, and conflict-of-interest statement. We will insert the verified declarations without changing the scientific content or results.

**Changes made:** Dedicated Author Contributions, Funding, and Conflict of Interest sections retained for final author-approved text. This is the only item requiring direct factual confirmation from all authors before submission.

### Comment R1-7: Improve conciseness and remove repeated implementation/audit details

**Response:** We agree. The previous source compiled an additional “Detailed Method and Structural Invariants” block after the conclusion, which repeated the main method, label-accounting, statistical, and audit discussion. We removed that block from the compiled main article and retained detailed operational material only as supplementary source. We also removed the sequential development-ablation paragraph from the main Results because the new controlled component experiments supersede it, consolidated statistical interpretation in the Experimental Design section, and rewrote the Discussion around the four research questions.

**Changes made:** Removed the compiled extended_sections input; shortened Results and Discussion; retained configuration, state-machine, performance, and archival details in appendices/supplementary material.

### Positive assessment

**Response:** We appreciate the reviewer’s recognition of the manuscript’s transparent limitations, failure analysis, and systems trade-off reporting. We retained the original worst-case cell and added the unfavorable all-module routing result rather than optimizing the narrative around positive findings. We believe this makes the revised contribution narrower but substantially more credible.

---

## Response to Reviewer 2

### Comment R2-1: The methodological contribution is not sufficiently isolated

**Response:** We agree and added the same 5,000-run routing/locality matrix and 225-run component study described in Responses R1-1 and R1-3. The revised manuscript no longer attributes the original integrated-system gain to the combined router. The evidence supports replay and performance gating, exact intervention control, and adaptation-side efficiency; it does not support predictive superiority of combined routing over all-module adaptation.

**Changes made:** Equal-Exposure Routing and Locality Controls; Component Isolation; RQ2 discussion; Abstract and Conclusion.

### Comment R2-2: Enforce equal label exposure for DASP and the primary comparator

**Response:** We implemented a common 512-label query contract for DASP, all routing/locality controls, and SSF. The exact queried indices and their digests are matched within every paired cell. The original 4,000-run result is retained but explicitly described as protocol-specific. The new equal-exposure result is reported separately and does not show robust DASP superiority over SSF.

**Changes made:** Problem Formulation; Methods and Fairness Controls; Equal-Exposure Routing and Locality Controls; Limitations.

### Comment R2-3: Quantify locality, test k=1/2/3, and add classifier-only adaptation

**Response:** We added all requested conditions and report both representation and total optimizer fractions. At k=1, k=2, and k=3, DASP opens 24.6%, 49.2%, and 75.4% of representation parameters. Because the shared classifier remains trainable, the corresponding total optimizer fractions are 85.7%, 90.4%, and 95.3%. Classifier-only adaptation opens 0% of representation parameters but 81.1% of total parameters. Inactive-module maximum change is zero in all audited repairs.

No tested k value significantly dominates combined k=3 after multiplicity correction. Combined k=3 is 0.001448 above classifier-only adaptation on average, but the corrected sign and Wilcoxon tests are not significant. We now use “representation-subspace-selective” or “structurally localized” and explicitly avoid describing k=3 as parameter-sparse.

**Changes made:** Equal-exposure table; Quantitative Locality table; RQ3 discussion; Abstract; Limitations.

### Comment R2-4: Test robustness to contiguous feature ordering

**Response:** We added 405 runs covering A1 combined routing, A6 all-module adaptation, and A9 classifier-only adaptation under three fixed random feature permutations. No condition shows a detectable loss. For A1, permuted-minus-original Macro-F1 is +0.000777 with a 95% cluster-bootstrap interval of −0.002580 to +0.003798. The corresponding A6 and A9 differences are −0.000069 and +0.000320. We interpret this as robustness to the tested orderings, not as evidence that the subspaces are semantically coherent.

**Changes made:** Quantitative Locality and Feature-Order Robustness subsection; RQ3 discussion; Limitations.

### Comment R2-5: Clarify pilot influence and add untouched validation

**Response:** We clarified the provenance of all P0 values. The four-module architecture, k=3, MMD scales, severity threshold, learning rate, repair epochs, and new-sample weight were inherited from the existing primary configuration. The equal-exposure pools, replay capacity, fixed routing mask, and random-routing namespace were locked before execution. Smoke and gate runs were inspected only for correctness, degenerate pools, invariant violations, runtime, and numerical failures; no P0 value was changed in response to predictive results.

We also agree that the original confirmation primarily shows robustness to new seeds within the same dataset and drift families. We now state that limitation explicitly and add the untouched natural chronological NF-ToN-IoT-v3 stream. Because this is one stream rather than an independent multi-site benchmark, we call it a limited external check rather than full external validation.

**Changes made:** Pilot and Confirmation Matrices; Reproducibility section; Natural Chronological Stream; Limitations; hyperparameter-provenance archive.

### Comment R2-6: Add cluster-aware uncertainty analysis

**Response:** We added cluster-aware analysis as the primary inferential layer for the new experiments. Seed/model cells are averaged within dataset–scenario clusters, yielding 50 P0 clusters and 15 P1 clusters. We use 20,000 cluster-bootstrap replicates, paired sign tests, Wilcoxon sensitivity tests, Holm correction within test families, and leave-one-dataset-out analyses. The original 500-cell tests remain only as within-matrix sensitivity analyses; we no longer emphasize their nominal p values as if the cells represented independent operational environments.

**Changes made:** Metrics and Statistical Analysis; Equal-Exposure and Component Results; RQ discussions; Statistical Audit appendix.

### Comment R2-7: Add an unmodified chronological experiment

**Response:** We added the timestamp-ordered NF-ToN-IoT-v3 experiment described in Response R1-4. It is reported without injected drift transformations and with descriptive rather than environment-level inferential claims.

**Changes made:** Unmodified Natural Chronological Stream subsection and table; RQ4 discussion; Limitations.

### Comment R2-8: Clarify the systems trade-off and intended operating regime

**Response:** We agree and removed broad language implying that DASP is simply “more efficient.” The revised paper distinguishes adaptation time, parameter-gradient work, inference throughput, and memory. In the original confirmation, DASP adapts faster than SSF but predicts substantially more slowly than flat MLP methods. The natural stream shows the same qualitative trade-off. We now state that DASP is intended for settings where bounded representation motion, adaptation latency, and auditability are more important than maximum inference throughput. A throughput-constrained deployment may prefer a flat or fused implementation.

**Changes made:** Abstract; Original-Protocol Adaptation Cost and Systems Performance; Natural Chronological Stream; RQ4 discussion; Limitations; Conclusion.

---

## Summary of the Revision

- Preserved the original 1,200-run pilot and 4,000-run eight-method confirmation as protocol-specific integrated-system evidence.
- Added 5,000 equal-label-exposure routing, k-sensitivity, all-module, classifier-only, and SSF runs.
- Added 225 component-isolation runs, 405 feature-permutation runs, and 12 natural-chronological runs.
- Added dataset–scenario cluster inference with 20,000 bootstrap replicates and Holm correction.
- Quantified representation and total trainable-parameter locality.
- Reframed DASP as an auditable representation-subspace-selective framework rather than a universally superior routing rule.
- Corrected and expanded repository, commit, environment, command, lock, and checksum information; the tagged revision archive remains to be uploaded before resubmission.
- Removed repeated detailed-method material from the compiled main article and consolidated the Discussion.
- Retained unfavorable and boundary results, including the all-module advantage, the heterogeneous equal-exposure SSF comparison, the worst original cell, and the inference-throughput penalty.

We thank the Editor and Reviewers again. The additional controls materially changed and strengthened the scientific interpretation. We believe the revised manuscript now distinguishes clearly among integrated effectiveness, component evidence, structural guarantees, and operational trade-offs.

Sincerely,

The Authors
