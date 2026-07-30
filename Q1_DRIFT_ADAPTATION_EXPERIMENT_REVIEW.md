# Cách các bài Q1 thực nghiệm khả năng thích nghi drift

Ngày tổng hợp: 2026-07-25

## Kết luận chính

Các bài mạnh không xem một điểm Macro-F1/accuracy cao trên toàn stream là đủ để
xác nhận “thích nghi drift”. Thiết kế thuyết phục thường ghép ba lớp bằng chứng:

1. **Drift có ground truth** trên stream tổng hợp hoặc bán tổng hợp để biết detector
   báo đúng, báo trễ hay báo giả.
2. **Stream thực theo thời gian** để kiểm tra tính thực tế, không shuffle và không
   dùng nhãn tương lai.
3. **Đối chứng cơ chế/ablation** để chứng minh phần thích nghi tạo ra lợi ích, thay
   vì lợi ích đến từ học online thông thường, replay, tăng compute hoặc cập nhật quá
   thường xuyên.

Do đó, thí nghiệm NF-UNSW-NB15-v3 hiện tại mới xác nhận được **stream recurrence
hợp lệ**. Nó chưa xác nhận cơ chế thích nghi drift của DASP-v12 vì trên 5 seed:
DASP-v12 có 0 trigger, 0 commit và kết quả trùng Experience Replay. Đây là một
kết quả âm có giá trị: nhánh PSLA/adaptation không tham gia vào dự đoán.

## Các thiết kế tiêu biểu

| Công trình | Cách tạo/quan sát drift | Cách đánh giá | Bài học trực tiếp cho DASP |
|---|---|---|---|
| [Adaptive Random Forests, Machine Learning 2017](https://doi.org/10.1007/s10994-017-5642-8) | Stream tổng hợp và thực; cả nhãn tức thời lẫn nhãn trễ | Test-then-train/prequential; accuracy, CPU, bộ nhớ | Phải khóa thời điểm nhãn xuất hiện và đo cả chi phí; dự đoán phải được chấm trước khi mẫu đó được dùng để học. |
| [ROSE, Machine Learning 2022](https://doi.org/10.1007/s10994-022-06168-x) | Bảy nhóm thí nghiệm: mất cân bằng tĩnh/động, noise drift, mẫu khó, stream thực | So với 30 ensemble; prequential Kappa/AUC; Friedman, Bonferroni-Dunn, Bayesian test; ablation từng thành phần | Stress test drift phải đồng thời thay đổi imbalance, noise và độ khó cục bộ; ablation là bắt buộc để gán công trạng cho memory, detector và background adaptation. |
| [OGMMF-VRD, IEEE TKDE](https://arxiv.org/abs/2102.05983) | 7 stream tổng hợp và 3 stream thực; tách virtual/real, abrupt/gradual/recurrent; thêm noise 5–20% | Accuracy-over-time, G-mean, runtime; Friedman–Nemenyi; phân tích cơ chế | Không gộp mọi drift thành một loại. Đường cong sụt giảm và thời gian phục hồi sau từng drift quan trọng hơn điểm trung bình toàn stream. |
| [Concept Drift Adaptation by Exploiting Drift Type, ACM TKDD 2024](https://doi.org/10.1145/3638777) | Dữ liệu tổng hợp có loại và thời điểm drift đã biết; dữ liệu thực không có ground truth drift | Tách thí nghiệm nhận diện loại/thời điểm drift khỏi thí nghiệm accuracy/F1 sau adaptation | Detector và adaptor phải được kiểm tra thành hai khối riêng. Một adaptor tốt không cứu được detector không kích hoạt. |
| [Concept-drift locality benchmark, Knowledge-Based Systems 2024](https://doi.org/10.1016/j.knosys.2024.111535) | 2.760 benchmark thay đổi số lớp bị ảnh hưởng và độ lớn drift, nhị phân lẫn đa lớp | So sánh 9 detector theo mức locality/magnitude | Detector toàn cục có thể bỏ sót drift chỉ tác động lớp hiếm hoặc một vùng feature. S4/local attack drift phải là test chính, không phải phụ. |
| [Delayed labelling evaluation, Data Mining and Knowledge Discovery 2020](https://doi.org/10.1007/s10618-019-00654-y) | 3 stream tổng hợp, 3 stream thực; mô phỏng verification latency khác nhau | Chấm dự đoán tại lúc mẫu đến và các thời điểm trước khi nhãn đến | “Nhãn trễ một cửa sổ” phải là một phần của protocol và metric; không được chấm theo trạng thái mô hình sau khi đã nhìn thấy nhãn của chính mẫu. |
| [K-means/SVM drift-aware network anomaly detection, Expert Systems with Applications 2022](https://doi.org/10.1016/j.eswa.2022.116510) | Testbed, NSL-KDD, CIDDS-2017; sliding windows; detector theo error và theo distribution | Accuracy, precision, recall, F1, Kappa, KL divergence; retrain khi kiểm định báo drift | IDS nên kết hợp tín hiệu thay đổi phân phối với hiệu năng theo lớp, nhưng điểm tổng hợp vẫn chưa đủ nếu không có drift ground truth và đối chứng “không adaptation”. |

Quartile phụ thuộc năm và category. Bảng trên dùng nhóm tạp chí thường được xếp
Q1 trong các category AI/ML/information systems liên quan; khi viết manuscript
cần ghi rõ hệ xếp hạng, năm và category thay vì chỉ ghi “Q1”.

## Mẫu giao thức thực nghiệm chung của bài mạnh

### 1. Hai loại stream bổ sung cho nhau

- **Synthetic/controlled:** biết chính xác drift point, loại drift, độ dài chuyển
  tiếp, magnitude và lớp bị ảnh hưởng. Dùng để đo detector.
- **Real chronological:** giữ nguyên thứ tự thời gian và quy tắc nhãn đến. Dùng để
  đo giá trị triển khai, nhưng thường không đủ để đánh giá precision/delay của
  detector vì drift ground truth không chắc chắn.
- **Semi-synthetic trên dữ liệu thực:** giữ feature và class statistics của dữ liệu
  thật nhưng ghép các regime theo lịch đã khóa. Đây là cầu nối hữu ích giữa hai
  loại trên, miễn là công bố rõ drift được tạo nhân tạo.

### 2. Test-then-train

Với cửa sổ \(t\):

1. mô hình dự đoán \(X_t\);
2. lưu prediction và chấm khi \(y_t\) hợp lệ xuất hiện;
3. chỉ sau đó mới cập nhật mô hình theo label-delay đã khóa.

Mọi normalization, feature selection, threshold calibration và candidate
selection cũng phải tuân theo causal order này.

### 3. Tách metric detector khỏi metric classifier

**Detector**

- Missed Detection Rate (MDR);
- Mean/median Time to Detection (MTD);
- Mean Time Between False Alarms (MTFA);
- precision/recall của trigger trong tolerance window;
- số warning, alarm và false commit.

**Adaptation**

- Macro-F1, balanced accuracy, G-mean và per-class recall theo thời gian;
- đáy hiệu năng ngay sau drift;
- số cửa sổ để quay lại 90% hoặc 95% mức trước drift;
- diện tích thiếu hụt sau drift (post-drift recovery deficit);
- chi phí update, gradient evaluations, memory và label budget;
- forgetting trên concept cũ và lợi ích khi concept tái xuất hiện.

Một benchmark gần đây chỉ ra rằng classifier accuracy/F1 dùng như proxy cho
detector có thể thiên vị detector báo nhiều, thậm chí cập nhật định kỳ ngây thơ
có thể thắng detector phức tạp. Vì vậy không được dùng cumulative Macro-F1 một
mình để kết luận detector tốt
([benchmark unsupervised drift detectors, 2025](https://doi.org/10.1007/s41060-024-00620-y)).

### 4. Đối chứng cơ chế

Ít nhất phải có:

- **Static/no update:** cho biết drift thật sự làm mô hình suy giảm bao nhiêu.
- **Online/Experience Replay:** cho biết học online thông thường đạt được gì.
- **DASP-v12 detector off:** cùng backbone, memory và budget nhưng vô hiệu nhánh
  drift.
- **DASP-v12 oracle trigger:** kích hoạt tại drift point thật; cho biết adaptor có
  tiềm năng khi detector đúng.
- **DASP-v12 nguyên trạng:** chuỗi detector → candidate → guard → commit hoàn chỉnh.
- **Ablation:** bỏ lần lượt memory refresh, layer conditioning, candidate
  class-balanced selection và recurrence retrieval.

So sánh oracle với nguyên trạng phân biệt lỗi detector và lỗi adaptor:

- oracle thắng nhưng nguyên trạng không thắng: lỗi ở detector/gating;
- cả oracle và nguyên trạng không thắng replay: lỗi ở candidate/adaptation;
- nguyên trạng trùng replay và không trigger: chưa hề thử nghiệm cơ chế drift.

## Protocol nên dùng cho DASP sau v12

### A. Controlled drift suite

Giữ các kịch bản hiện tại nhưng báo cáo theo episode:

- S0: không drift, đo false alarm/false commit;
- S1: first abrupt drift;
- S2: gradual/incremental drift;
- S3: \(A \rightarrow B \rightarrow A\), tách **first drift** và **recurrence**;
- S4: local/minority-class drift;
- S5: virtual drift so với real drift;
- thêm lưới nhỏ cho magnitude, số lớp bị ảnh hưởng, noise, imbalance và label
  delay.

Với S3, báo cáo riêng:

\[
\Delta_{\text{first}} =
F1_{\text{v12, first }k} - F1_{\text{ER, first }k}
\]

\[
\Delta_{\text{rec}} =
F1_{\text{v12, recurrence }k} - F1_{\text{ER, recurrence }k}
\]

và recurrence speed-up so với lần gặp concept đó đầu tiên. Không được dùng điểm
recurrence gần 0,99 để che việc first drift còn thua SSF.

### B. NF-UNSW-NB15-v3

Stream hiện tại vẫn nên giữ vì chronological và recurrence tự nhiên hợp lệ.
Nhưng phải thêm ba audit:

1. **Detector audit:** severity từng layer, threshold, warning/alarm, delay so với
   ranh giới A1→B và B→A2.
2. **Adaptation audit:** candidate nào được train, guard nào từ chối, có commit
   hay rollback và lượng dữ liệu/compute đã dùng.
3. **Causal audit:** v12 nguyên trạng, detector-off/replay và oracle-trigger chạy
   trên cùng seed, initialization, batch, nhãn và budget.

Do phase B không có attack, Macro-F1 từng cửa sổ có thể trở thành đại lượng khó
diễn giải. Cần báo cáo thêm false-positive rate/specificity ở B, attack recall ở
A1/A2, cumulative confusion matrix theo episode và sample-weighted Macro-F1.

Kết quả hiện tại nên được viết đúng mức:

> Trên natural NF-UNSW recurrence stream, DASP-v12 đạt hiệu năng dự đoán cao,
> nhưng detector không kích hoạt; vì vậy kết quả xác nhận replay backbone, không
> xác nhận PSLA/drift adaptation.

### C. Fairness và thống kê

- Khóa seed phát triển; không tiếp tục chỉnh threshold trên 1301–1305.
- Chỉ dùng seed xác nhận mới sau khi protocol, generator S4 và implementation SSF
  đã được khóa.
- Dùng paired seeds; bootstrap CI cho chênh lệch episode; Wilcoxon có Holm cho
  các đối chiếu đã đăng ký; Friedman/post-hoc khi so nhiều thuật toán.
- Không coi các cửa sổ trong cùng một stream là các mẫu độc lập.
- Báo win/tie/loss, effect size và confidence interval, không chỉ p-value.
- Cân bằng label budget, update opportunities, backbone capacity và compute; nếu
  không thể cân bằng, báo Pareto performance–cost.

## Tiêu chí tối thiểu để nói “đã xác nhận thích nghi drift”

Một claim cơ chế của v12 chỉ hợp lệ khi đồng thời có:

1. Drift ground truth hoặc event boundary đã khóa.
2. Trigger đúng xảy ra ở đa số seed và false trigger trên S0 được kiểm soát.
3. Có candidate/commit thật, không chỉ warning log.
4. V12 hơn detector-off/Experience Replay trong early recovery với CI/effect size
   đã đăng ký.
5. Oracle-trigger cho thấy adaptor có ích; khoảng cách oracle–v12 giải thích được.
6. Kết quả giữ được trên ít nhất một stream thực chronological với label-delay
   thực tế.

Nếu thiếu mục 2 hoặc 3, chỉ được kết luận về chất lượng stream hoặc backbone,
không được kết luận về drift adaptation.

## Hạn chế của bằng chứng

- Nhiều stream thực không có ground truth drift; metric detector khi đó chỉ là
  proxy.
- Một số bài rất rộng nhưng vẫn dùng single run hoặc cách tổng hợp seed chưa lý
  tưởng; “đăng Q1” không tự động làm thiết kế hoàn hảo.
- Benchmark tree/ensemble không hoàn toàn đồng cấu với neural continual learning;
  comparator phải được cân bằng backbone, nhãn và compute trước khi diễn giải.
- Kết quả trên NF-UNSW nhị phân không tự động chuyển sang multiclass attack-family
  drift; cần một confirmation riêng cho bài toán đa lớp.

## Chiến lược tìm kiếm và nguồn

Tìm trên Springer, ACM, IEEE/author manuscripts và ScienceDirect bằng các nhóm từ
khóa: “concept drift adaptation”, “data stream prequential evaluation”,
“delayed labels”, “recurring drift”, “drift locality”, “imbalanced drift” và
“network intrusion detection concept drift”. Ưu tiên bài phương pháp gốc,
publisher page/full manuscript, thiết kế có synthetic ground truth, real stream,
ablation, statistical testing và mã nguồn. Bài không rõ venue hoặc chỉ mô tả
accuracy tĩnh được dùng làm bối cảnh, không dùng làm chuẩn thiết kế.

## AI disclosure

Tài liệu này được tổng hợp với hỗ trợ của AI. Các nhận định phương pháp được đối
chiếu với trang nhà xuất bản hoặc bản thảo của tác giả; quartile cần được kiểm tra
lại theo năm/category chính thức khi đưa vào manuscript.
