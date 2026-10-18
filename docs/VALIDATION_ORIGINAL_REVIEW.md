# Kiểm chứng bản DASP cải tiến — 18/10/2026

Các kết quả dưới đây xác nhận code và đường chạy. Chưa có full benchmark hay chứng minh chất lượng phù hợp Q1.

## Đã chạy

- **169/169 pytest pass**, không bỏ qua test module. Có21case mới cho original-DASP và analysis; bao gồm gate, loại inactive gradient inputs, conflict accounting, legacy equivalence khi giữ gate, query causality, parameter-matched flat controls, cache corruption/config mismatch và checkpoint roundtrip. Có5warnings sklearn ở one-class metrics; không phải failures.
- **23/23 CUDA smoke runs**, gồm16DASP controls và7comparators gốc. Cùng schema main, một synthetic fixture nhỏ để kiểm tra luồng. Đã thử `--resume` cho cả23runs: reuse completed outputs với đúng hashes, không train lại.
- **2CUDA runs với dữ liệu thật đã dùng trước đây**: NF-BoT-IoT-v3/S0,stream2001/model91; cấu hình original memory1024/replay512, cùng prepared stream và hyperparameters. Gate sửa mean-window Macro-F1≈0.936009; gate legacy≈0.933468. Đây là một paired diagnostic, không độc lập/external validation. CPU diagnostic trước đó cho≈0.936009 và≈0.934049; không cam kết bitwise equality giữa CPU và GPU.
- **Primary plan4000runs** được tạo và test đúng danh sách5datasets/10scenarios/8methods/5stream seeds/2model seeds. Chưa chạy toàn ma trận.
- Base config, original protocol grid, baseline/model/data/revision/metrics/memory/drift sources được giữ nguyên theo lock hash. Chỉ đường original modular_dasp dùng corrections mới; measurement boundary chung không thay learning logic của đối chứng.

## Môi trường đã kiểm tra

Windows, Python3.12.7, PyTorch2.6.0+cu124, CUDA build12.4, RTX3050Laptop6GB. NumPy2.4.6 và SciPy1.18.1 của environment hiện có khác historical lock; ENVIRONMENT.json lưu phiên bản thực. File requirements Linux là pins tách từ lock lịch sử, chưa phải environment Linux đã test. Cần chạy cùng smoke/preflight trên server và lưu pip freeze trước full experiment.

Các junction dữ liệu trong worktree chỉ trỏ read-only theo workflow tới prepared streams có sẵn ở workspace gốc; không có dữ liệu trong ZIP. Hai test cũ ban đầu thiếu các đường dẫn ignored trong clean worktree đã pass khi đường dẫn input được nối lại. Không sửa/xóa test để che lỗi.

## Phạm vi đo lường và giới hạn

- Counters mới: post-warmup adaptation training+conflict. Không bao gồm warmup và không là FLOPs. Baseline counters giữ nghĩa lịch sử; không so trực tiếp nếu scope khác.
- Full-cycle timing chung gồm drift, prediction, metrics, delayed update và reference refresh. Warmup và terminal updates được phân biệt. Locality snapshots/diagnostics có overhead và nằm trong update time; dùng root riêng với `--no-locality-check` khi cần timing sensitivity.
- `peak_rss_bytes` lịch sử là maximum ở các phase quan sát, không phải continuous/OS true peak. `memory_payload_bytes` là replay payload, không phải tổng resident model/optimizer/reference state. Không tuyên bố total-memory superiority từ hai trường đó.
- Main label protocol vẫn khác giữa methods. DASP báo exposure của repair selection/gate, không chỉ training count; fairness stage mới là capped-query comparison. Warmup trong thiết kế cũ cấp labels toàn warmup để seed memory; các headline supervision totals cần đọc scope post-warmup và thêm warmup riêng khi báo tổng.
- Native SSF/CDDA/RepShield reproduction chưa được xác minh; baseline bugs ngoài phạm vi DASP không được sửa ở nhánh này. CUDA smoke chỉ xác nhận chúng chạy, không chứng nhận correctness/fidelity.
- Routing masks có thể khác nhưng F1 giống nhau trên fixture nhỏ. Không dùng điều đó làm bằng chứng router tương đương/tốt hơn trên dữ liệu thực.
- Natural/permutation/components/full routing/fairness matrices chưa chạy hết. Không có Linux server access hoặc upload/deploy ở lượt này.

## Artefacts

`outputs_original_review_validation/release_suite.xml`, `outputs_original_review_gpu_release/smoke_analysis.json`, `outputs_original_review_gate_release/gate_analysis.json`, `outputs_original_review_primary_plan/primary_plan.json`. ZIP kèm XML/analysis/environment nhỏ trong validation/, không kèm raw/prepared data.
