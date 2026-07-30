# Protocol thực nghiệm Selective-DASP trên năm IDS datasets

**Phiên bản:** 1.0  
**Ngày khóa thiết kế:** 2026-07-26  
**Trạng thái:** khóa phương pháp trước confirmation; chỉ được sửa lỗi triển khai
đã được test chứng minh, không chỉnh hyperparameter theo kết quả confirmation.

## 1. Mục tiêu và giới hạn kết luận

Mục tiêu chính là kiểm định liệu **Selective-DASP** có thích nghi với concept
drift tốt hơn các đối chứng hợp lý trong IDS streaming hay không. Một điểm
Macro-F1 trung bình cao trên toàn stream không đủ để xác nhận cơ chế thích nghi.
Kết luận chính chỉ được đưa ra khi đồng thời có bằng chứng về:

1. detector phát hiện đúng các drift episode đã biết;
2. nhánh thích nghi thực sự tạo candidate và commit;
3. cải thiện early post-drift Macro-F1 và recovery so với đối chứng không dùng
   detector và các baseline mạnh;
4. chi phí inference, update, memory và label budget chấp nhận được;
5. kết quả lặp lại trên nhiều dataset và data stream độc lập.

Nếu detector không trigger hoặc không có adaptive commit, kết quả chỉ xác nhận
backbone/replay, không xác nhận drift adaptation. Kết quả âm vẫn được báo cáo;
không tiếp tục đổi threshold hoặc seed cho đến khi có kết quả thuận lợi.

## 2. Câu hỏi nghiên cứu và giả thuyết

### RQ1 — Khả năng phục hồi sớm

Selective-DASP có Macro-F1 trung bình trong năm cửa sổ đầu sau drift cao hơn SSF
budget-matched hay không?

### RQ2 — An toàn lớp tấn công

Lợi ích Macro-F1 có đạt được mà không làm Attack Recall giảm quá 0,01 so với SSF
hay không?

### RQ3 — Giá trị của detector và adaptor

Selective-DASP có tốt hơn phiên bản detector-off và tiến gần oracle-trigger hay
không? Oracle chỉ là diagnostic upper bound, không phải comparator triển khai.

### RQ4 — Recurrence và forgetting

Trong \(A\rightarrow B\rightarrow A'\), Selective-DASP có phục hồi nhanh hơn khi
concept A xuất hiện lại và có forgetting thấp hơn Replay/SSF hay không?

### RQ5 — Hiệu quả vận hành

Selective-DASP có duy trì latency, throughput, memory, label use và update cost
phù hợp xử lý streaming hay không?

## 3. Datasets và provenance

### 3.1 Năm datasets

1. NF-UNSW-NB15-v3
2. NF-CSE-CIC-IDS2018-v3
3. NF-ToN-IoT-v3
4. NF-BoT-IoT-v3
5. LUFlow

Bốn NF-v3 dùng cùng schema 53 NetFlow-v3 features. File NF-ToN-IoT-v3 và
NF-BoT-IoT-v3 được lấy từ kho Research Data Manager của University of
Queensland; SHA-1 sau giải nén phải khớp BagIt manifest:

| File | Số flow kiểm toán | SHA-1 |
|---|---:|---|
| NF-ToN-IoT-v3.csv | 27,520,260 | `24854ca3072ab7c2ade9ebfb202345a666a94cfe` |
| NF-BoT-IoT-v3.csv | 16,933,808 | `357cc7ffe46ef57ba8193a58d7a66eff5cf44db8` |

Manifest và checksum của tất cả dữ liệu đầu vào phải được ghi vào material
passport của lần chạy. Không thay file giữa development và confirmation.

### 3.2 Nhiệm vụ

- **Primary:** binary classification, Benign và Attack.
- **Secondary:** multiclass theo attack family khi taxonomy của dataset hỗ trợ.

LUFlow chỉ có nhãn `benign`, `malicious`, `outlier`, không có taxonomy attack
family tương thích bốn NF-v3. Vì vậy LUFlow là external chronological track
chính; không dùng LUFlow để tạo S4 attack-family drift giả.

## 4. Hai track bổ sung cho nhau

### 4.1 Controlled semi-synthetic track — bằng chứng cơ chế chính

Áp dụng trên bốn NF-v3. Feature vectors và labels đều là flow thật; chỉ lịch
ghép regime được kiểm soát. Mỗi stream có 36 cửa sổ, mỗi cửa sổ 4.096 flow:

- W0–W3: warm-up, không tính vào headline metrics;
- W4–W11: concept A ổn định;
- W12–W23: vùng concept B/chuyển tiếp;
- W24–W35: concept cuối.

Không flow nào lặp lại trong một stream. Với S3, A′ dùng flow mới thuộc cùng
attack-family composition với A; tuyệt đối không replay lại flow của A.

Tỷ lệ Attack được giữ cố định qua mọi cửa sổ trong cùng dataset và scenario để
drift không bị trộn với label-prior shift. Tỷ lệ primary là prevalence đã kiểm
toán của dataset, giới hạn trong \([0.05,0.95]\):

| Dataset | Attack rate khóa |
|---|---:|
| NF-UNSW-NB15-v3 | 0.05399 |
| NF-CSE-CIC-IDS2018-v3 | 0.12930 |
| NF-ToN-IoT-v3 | 0.38982 |
| NF-BoT-IoT-v3 | 0.95000 |

Mỗi cửa sổ phải đạt tỷ lệ đã khóa trong sai số làm tròn tối đa một flow.
Sensitivity analysis dùng prevalence 0.05, 0.30 và 0.70 chỉ trên dataset có đủ
pool không trùng; không thay thế primary analysis.

### 4.2 Natural chronological track — bằng chứng external validity

Mỗi dataset tạo một **bounded exact chronological prefix** gồm 96 cửa sổ × 4.096
flow theo `FLOW_START_MILLISECONDS` (NF-v3) hoặc `time_start` (LUFlow).
Generator quét toàn nguồn để lấy đúng 393.216 flow sớm nhất, rồi sort ổn định
bằng timestamp và original row id. Không được gọi thứ tự vật lý của CSV là
chronology khi chưa audit. Track này là prefix liên tục có giới hạn, không đại
diện toàn bộ timeline; giới hạn đó phải xuất hiện trong Methods và Limitations.
Full-timeline run, nếu có đủ compute, là supplementary.

Chỉ warm-up một lần. Không reset model, scaler, detector hoặc memory giữa Early,
Middle và Late. Giữ nguyên tỷ lệ lớp, burst, missingness và thứ tự tự nhiên.
Natural track không có drift ground truth đáng tin cậy nên chỉ dùng để đánh giá
prequential classification và chi phí; không báo detector precision/delay như
sự thật nền.

## 5. Controlled drift generator

### 5.1 Khái niệm A và B

Concept được định nghĩa trước bằng các attack-family có đủ mẫu:

| Dataset | Concept A | Concept B | S4 stable/local pools |
|---|---|---|---|
| NF-UNSW-NB15-v3 | Exploits + Fuzzers (50/50) | Generic + Reconnaissance + DoS (45/45/10) | stable=Exploits/Fuzzers/Generic; local A=DoS; local B=Reconnaissance |
| NF-CSE-CIC-IDS2018-v3 | DDoS/DoS families | Brute-force + Bot + Infilteration | stable=HOIC; local A=Hulk/GoldenEye; local B=FTP/SSH brute-force |
| NF-ToN-IoT-v3 | ddos + dos | password + scanning + injection | stable=ddos; local A=dos; local B=scanning |
| NF-BoT-IoT-v3 | DDoS | DoS | stable=DoS; local A=Reconnaissance; local B=DDoS |

Sampling trong một concept là stratified theo family và time quantile. Pool
Benign, A, B và A′ được tách theo original row id trước khi tạo cửa sổ. Generator
ghi manifest gồm dataset hash, row ids/hash của từng pool, seed, class counts,
family counts, timestamp range và overlap audit.

### 5.2 Năm canonical scenarios

1. **S0 Stationary:** A trong W0–W35.
2. **S1 Abrupt:** A trong W0–W11; B trong W12–W35.
3. **S2 Gradual:** A trong W0–W11; tỷ lệ B ở W12–W23 lần lượt
   \(1/12,2/12,\ldots,12/12\); B trong W24–W35.
4. **S3 Recurring:** A trong W0–W11; B trong W12–W23; A′ trong W24–W35.
5. **S4 Local/minority:** phần lớn attack pool giữ ổn định; đúng 5% attack flows
   chuyển từ local-A sang local-B tại W12, các phần còn lại không đổi.

S4 thay đổi 5% của lớp Attack, không phải 5% toàn stream. Nếu một source pool
không đủ sau khi chia data seeds, generator dừng với lỗi thay vì sampling có
replacement hoặc âm thầm đổi tỷ lệ.

### 5.3 S5 high-severity adversarial stress track

S5 là secondary stress track, không thuộc primary S0–S4 family:

- W0–W11 dùng concept A;
- W12–W35 dùng `B_evasive`;
- tỷ lệ Attack/Benign và family weights của concept B giữ nguyên;
- chỉ dùng flow thật, không sửa feature, không đổi nhãn và không sampling có
  replacement.

`B_evasive` được tạo bằng một surrogate ensemble cố định gồm Logistic
Regression, Random Forest và Extra Trees. Ensemble học trên generator-design
pool Benign/A không trùng bất kỳ stream seed nào. Trong từng concept-B family
và từng time quantile, generator ưu tiên các Attack flow có median
\(P(\mathrm{Benign}\mid x)\) cao nhất. Một probe ensemble khác, học trên
generator-audit split không trùng, kiểm tra severity.

Một S5 artifact chỉ hợp lệ khi đồng thời đạt:

- consensus evasion rate ít nhất 0,70;
- median Attack-Recall drop ít nhất 0,30;
- median Macro-F1 drop ít nhất 0,20;
- so với một control B ngẫu nhiên có cùng family weights, attack prevalence và
  time-quantile allocation, evasion uplift ít nhất 0,10, Attack-Recall-drop
  uplift ít nhất 0,10 và Macro-F1-drop uplift ít nhất 0,05;
- tập `B_evasive` và matched-random control không dùng chung source row;
- mọi invariant về prevalence, source-row overlap và preprocessing vẫn đạt.

Nếu gate thất bại, policy mặc định là dừng và không gọi stream đó là
high-severity. S5 được báo riêng như **high-severity adversarial
class-conditional drift**, không gọi là full-concept drift khi chưa có bằng
chứng \(p(y\mid x)\) thay đổi trên toàn input space.

### 5.4 Construct validity

Các scenario là **class-conditional composition drift bán tổng hợp**. Trước khi
gọi đó là “real concept drift”, phải kiểm tra:

- \(p(y)\) ổn định đúng tolerance;
- phân phối \(p(x\mid y=\mathrm{Attack})\) thay đổi giữa A và B;
- probe classifier học trên A suy giảm trên B và ngược lại;
- S0 không có thay đổi có hệ thống ngoài sampling noise.

Nếu chỉ \(p(x)\) thay đổi nhưng \(p(y\mid x)\) và boundary không thay đổi đáng
kể, episode được ghi là virtual/covariate shift và không dùng để xác nhận khả
năng thích nghi real drift.

## 6. Causal online protocol

Label delay khóa ở một cửa sổ. Với cửa sổ \(t\):

1. nhận \(X_t\);
2. preprocessing bằng scaler đã fit duy nhất trên warm-up;
3. mô hình trạng thái \(M_t\) dự đoán \(X_t\);
4. ghi prediction và inference latency;
5. detector chỉ dùng \(X_{\le t}\), prediction và state hợp lệ tại \(t\);
6. nhận nhãn \(y_{t-1}\);
7. cập nhật bằng \(X_{t-1},y_{t-1}\) và tối đa feature của \(X_t\) chưa gán nhãn;
8. candidate mới chỉ được dùng từ cửa sổ \(t+1\).

Nhãn \(y_t\) không được dùng cho drift decision, replay selection, candidate
training hoặc guard tại thời điểm \(t\). Oracle-trigger chỉ được nhìn lịch drift
đã đăng ký, không được nhìn nhãn hiện tại.

Scaler, imputation median và feature schema chỉ được fit/khóa trên W0–W3.
Không feature selection hoặc normalization nào dùng future windows.

## 7. Methods và fairness

### 7.1 Comparator set

1. Frozen model;
2. Periodic fine-tuning;
3. Online MLP;
4. Experience Replay;
5. SSF faithful mode;
6. SSF budget-matched mode — comparator chính;
7. RepShield;
8. Selective-DASP detector-off;
9. Selective-DASP oracle-trigger;
10. Selective-DASP proposed;
11. DASP-v12 whole-update upper bound — diagnostic, không thỏa ràng buộc
    selective-only.

Tree/stream baselines như ARF/HAT được báo ở bảng heterogeneous-baseline riêng,
không diễn giải như ablation đồng cấu.

### 7.2 Fairness tiers

**Tier A, budget-matched:** cùng backbone, initialization, warm-up, memory
capacity, label budget, delayed labels và trần gradient evaluations. Đây là
so sánh causal chính.

Primary protocol giả định toàn bộ nhãn của cửa sổ trước xuất hiện sau một cửa
sổ; guard subset vì vậy có thể chấm candidate cho mọi method. `label budget=256`
được định nghĩa chính xác là số ground-truth samples tối đa được dùng để
**huấn luyện hoặc chèn vào memory mỗi cửa sổ**, không phải active-label
acquisition budget. Mọi neural comparator dùng cùng deterministic
class-balanced subset 256; warm-up là ngoại lệ được khai báo và dùng toàn bộ.

**Tier B, faithful implementation:** dùng hyperparameter/cơ chế theo paper hoặc
official code, kể cả pseudo-labels nếu có. Khác biệt label/compute được công bố
và không thay Tier A.

Mỗi run ghi: số ground-truth labels, pseudo-labels, sample updates, gradient
evaluations, update opportunities, parameter groups được đổi, memory peak và
thời gian update. SSF phải ghi riêng ground-truth/pseudo insertions.

### 7.3 Ràng buộc Selective-DASP

- không candidate global/uniform whole-model update;
- `global_update_count == 0` trong mọi run;
- non-drift window chỉ được lightweight layer-conditioned refresh đã khóa;
- strong PSLA chỉ chạy khi detector hoặc delayed-performance trigger hợp lệ;
- replay từ current unlabeled batch chỉ dùng feature similarity, không dùng
  nhãn hiện tại;
- candidate được xếp theo Macro-F1, minimum/per-class recall và guard an toàn,
  không chỉ cross-entropy.

## 8. Metrics

### 8.1 Primary endpoints

Với drift onset \(t_d\):

\[
\mathrm{EarlyMacroF1@5} =
\frac{1}{5}\sum_{t=t_d}^{t_d+4}\mathrm{MacroF1}_t
\]

\[
\mathrm{EarlyAttackRecall@5} =
\frac{1}{5}\sum_{t=t_d}^{t_d+4}\mathrm{Recall}_{Attack,t}
\]

Trong S3 báo riêng:

- first drift: \(t_d=12\), A→B;
- recurrence: \(t_r=24\), B→A′.

Không gộp hai episode để che một first-drift failure.

### 8.2 Recovery

Reference performance là median Macro-F1 của W7–W11. Recovery threshold bằng
90% reference. Recovery time là số cửa sổ từ onset đến khi đạt threshold trong
hai cửa sổ liên tiếp. Run không recovery trước cuối episode được right-censor,
không gán bằng 0.

\[
\mathrm{RecoveryDeficit@5} =
\sum_{t=t_d}^{t_d+4}
\max(0,\;0.9F1_{\mathrm{ref}}-F1_t)
\]

Báo thêm minimum post-drift Macro-F1, area under deficit curve, recurrence gain,
old-concept forgetting, balanced accuracy, specificity/FPR và confusion matrix
theo episode.

Macro-F1 dùng tập nhãn cố định của task. Lớp vắng trong một cửa sổ không bị âm
thầm loại khỏi phép trung bình; metric theo episode/cumulative confusion matrix
được dùng để tránh diễn giải sai cửa sổ chỉ có một lớp.

### 8.3 Detector

Một alarm chỉ match một drift episode và một drift chỉ match một alarm.

- abrupt/local/recurrence tolerance: onset đến onset+5;
- gradual tolerance: W12 đến W25;
- detection precision/recall;
- missed drift;
- false alarms và mean time between false alarms trên S0;
- detection delay;
- warning, alarm, candidate, commit và rollback counts.

Natural track không báo các metric này nếu không có annotation độc lập.

### 8.4 Latency và tài nguyên

Đo riêng:

- preprocessing;
- model forward;
- drift scoring;
- adaptation decision/guard;
- update;
- end-to-end prediction path.

Sau warm-up thiết bị, báo median, p95, p99 latency theo flow và theo window,
throughput flow/s, peak RSS/VRAM, serialized model size và memory-buffer size.
CPU/GPU, số thread, batch size, precision, OS và phiên bản thư viện phải được
khóa. Synchronize accelerator trước/sau timer nếu dùng GPU.

Điều kiện vận hành primary: p95 prediction-decision latency nhỏ hơn inter-arrival
budget đã khai báo và throughput ít nhất 1,2 lần target input rate. Nếu chưa có
deployment target độc lập, chỉ báo raw latency/throughput, không tuyên bố
“real-time”.

## 9. Seeds, pairing và run matrix

- 5 **data stream seeds** tạo năm stream không trùng flow trong cùng dataset;
- 2 **model seeds** lồng trong mỗi fixed stream;
- tổng 10 paired runs cho mỗi dataset × scenario × method;
- mọi method dùng đúng cùng stream manifest và model initialization seed.

Data seed và model seed là hai trường riêng. Không cộng chúng thành một seed duy
nhất. Natural chronology chỉ có một temporal realization cho mỗi dataset; model
seeds không được gọi là các temporal replicates.

Primary controlled matrix có 4 NF-v3 × 5 scenarios × 10 paired runs × 10 methods
= 2.000 runs. DASP-v12 upper bound và heterogeneous baselines có thể chạy trong
supplementary matrix riêng để không làm thay đổi primary family of tests.

S5 secondary matrix có 4 NF-v3 × 1 scenario × 10 paired runs × 10 methods =
400 runs. S5 không được gộp vào primary S1–S4 average hoặc dùng để thay đổi
success criterion sau khi đã xem kết quả.

## 10. Development, freeze và confirmation

1. **Unit/integration tests:** causal order, label delay, pool overlap, S4,
   SSF budget, global-update invariant, latency fields.
2. **Debug seeds:** chỉ để tìm lỗi runtime; không dùng trong bảng kết quả.
3. **Development seeds:** dùng để chọn một cấu hình duy nhất. Seed 1301–1305 và
   các block đã dùng cho v12–v15 không được tái sử dụng để tinh chỉnh.
4. Tạo resolved configuration, stream manifests, code snapshot hash và
   preregistration.
5. **Confirmation seeds mới:** chạy một lần theo matrix đã khóa.
6. Mọi sửa lỗi sau khi nhìn confirmation phải được ghi changelog, chạy lại toàn
   bộ methods bị ảnh hưởng và gọi kết quả là revised confirmation.

Không chọn seed mới, bỏ seed xấu, đổi threshold hoặc thay success criterion sau
khi xem outcome.

## 11. Statistical analysis

Đơn vị độc lập là stream, không phải cửa sổ. Phân cấp:

dataset → scenario/episode → data seed → model seed.

Primary contrast là paired difference:

\[
\Delta = \mathrm{EarlyMacroF1@5}_{DASP}
- \mathrm{EarlyMacroF1@5}_{SSF\_budget}
\]

Hierarchical paired bootstrap resample dataset, data stream trong dataset và
model seed trong stream; báo point estimate, 95% CI, win/tie/loss và raw paired
values. Primary superiority đạt khi CI 95% của \(\Delta\) hoàn toàn lớn hơn 0.

Attack Recall dùng non-inferiority margin \(-0.01\): cận dưới 95% CI của
\(\Delta_{\mathrm{Recall}}\) phải lớn hơn \(-0.01\).

Phép sign-flip hai phía ở cấp independent dataset--stream cluster là
sensitivity analysis bắt buộc; dùng exact enumeration khi số cluster không quá
16 và Monte Carlo đã khóa seed khi lớn hơn. Các so sánh DASP với toàn bộ
baseline ngoài primary contrast được hiệu chỉnh Holm riêng theo từng metric.
Wilcoxon signed-rank/Friedman chỉ dùng như sensitivity bổ sung cho family đã
đăng ký. Không thực hiện test riêng cho từng cửa sổ như các quan sát độc lập.
Báo effect size, CI, số independent clusters, raw và adjusted p-value; CI
bootstrap không được diễn giải như bằng chứng quyết định khi exact
stream-cluster test không đạt.

## 12. Điều kiện kết luận

Chỉ kết luận Selective-DASP thích nghi drift tốt hơn khi:

1. EarlyMacroF1@5 cao hơn SSF budget-matched với hierarchical-bootstrap 95% CI
   > 0 trên controlled S1–S4;
2. hiệu ứng dương trên ít nhất 4/5 datasets xét trên toàn bộ bằng chứng, trong đó
   controlled-mechanism claim dựa trên bốn NF-v3;
3. Attack Recall đạt non-inferiority margin -0,01;
4. recovery tốt hơn Experience Replay và SSF;
5. tốt hơn detector-off;
6. oracle-trigger có ích và proposed thu hẹp đáng kể oracle gap;
7. S0 có false-alarm/false-commit thấp;
8. S3 first drift và recurrence đều không thất bại; forgetting thấp;
9. có adaptive commits thật ở đa số drift episodes;
10. `global_update_count == 0` trong mọi run;
11. latency, throughput, memory và label use được báo đầy đủ và không vượt budget
    đã khóa.

Nếu chỉ một số scenario/dataset đạt, kết luận phải giới hạn đúng phạm vi đó.
Nếu result không đạt gate, LaTeX chỉ được cập nhật như kết quả âm/development,
không dùng ngôn ngữ “outperforms” tổng quát.

## 13. Hình và bảng bắt buộc

1. Stream timeline A→B→A′ và causal label-delay timeline;
2. Macro-F1 theo cửa sổ với 95% paired uncertainty band;
3. Attack Recall theo cửa sổ;
4. episode recovery curves tách S3 first/recurrence;
5. detector raster: ground-truth onset, warning, alarm, commit;
6. forest plot paired EarlyMacroF1@5 theo dataset/scenario;
7. performance–latency và performance–memory Pareto plot;
8. ablation plot detector-off/proposed/oracle;
9. table dataset passport và attack-family composition;
10. table primary statistics, costs, labels và invariant audits.

Hình manuscript xuất PDF/SVG vector và PNG kiểm tra. Trục, đơn vị, n, CI và
định nghĩa episode phải ghi trong caption; không dùng bar chart chỉ có mean mà
không có paired uncertainty.

## 14. Reproducibility và integrity artifacts

Mỗi run lưu:

- raw per-window predictions/metrics;
- resolved YAML/JSON config;
- dataset checksum và stream manifest;
- data seed, model seed, method và code snapshot hash;
- detector/candidate/guard/commit event log;
- label/pseudo-label/sample/gradient budgets;
- latency samples và environment metadata;
- summary được tạo lại từ raw windows.

Analyzer phải kiểm tra số run kỳ vọng, duplicate/missing cells, pairing, NaN,
overlap, current-label leakage, SSF budget và global-update invariant trước khi
tạo bảng/hình. Manuscript chỉ đọc từ frozen analysis outputs, không chép số thủ
công.

## 15. Nguồn phương pháp chính

- UQ NetFlow-v3 datasets: <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>
- SSF, IEEE INFOCOM 2025: <https://arxiv.org/abs/2412.16264>
- IWMS/label delay, NeurIPS 2024:
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/d8f5f134febb4bd74d8f79e338de382c-Abstract-Conference.html>
- Drift-type adaptation, ACM TKDD 2024: <https://doi.org/10.1145/3638777>
- Drift-locality benchmark, Knowledge-Based Systems 2024:
  <https://doi.org/10.1016/j.knosys.2024.111535>
- Unsupervised detector benchmark, 2025:
  <https://doi.org/10.1007/s41060-024-00620-y>

## 16. AI disclosure

Thiết kế, mã và manuscript có hỗ trợ của AI. Dataset provenance, checksum,
resolved configs, raw outputs, thống kê và mọi số liệu trong bài phải được kiểm
tra bằng artifact có thể tái tạo. AI không được dùng để tạo, sửa hoặc lựa chọn
kết quả nhằm làm đẹp kết luận.
