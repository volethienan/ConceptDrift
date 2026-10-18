# DASP cải tiến từ bản gốc theo phản biện

Nhánh `codex/dasp-original-review-fixes`, nền `c7abfb18c34cd374211e1f198d8d35bdcac42578`.
Phiên bản thuật toán: `modular-dasp-original-review-20261018-v1`.

Mục tiêu là sửa và cải tiến **DASP gốc** theo nhận xét, giữ thiết kế thực nghiệm chính. Không đưa A1 hoặc các fast-adapter v3/v4 vào thay phương pháp đề xuất.

## Phần giữ cố định và phần được cải tiến

| Giữ nguyên ma trận chính | Thực hiện |
|---|---|
| 5 dataset,10 scenarios,8 methods | Import nguyên danh sách từ run_modular_dasp_protocol.py |
| stream seeds2001,2002,2003,2101,2102; model seeds91,92 | 4.000 run, không trộn record pilot cũ |
| Modular backbone4modules×64; head[128,64] | Giữ nguyên base config |
| Memory1024, replay512, new weighting3 | Không nâng thành5000 như A1 |
| Native routing, plasticity theo drift/conflict; background.25 | Giữ cơ chế làm nền, bổ sung đối chứng để kiểm tra đóng góp |
| Dự đoán trước, nhãn trễ1window, warmup và prepared streams cũ | Không chia lại/shuffle/chuẩn hóa lại dữ liệu |
| Background consolidation và repair2/3epochs | Không biến thành skip-update policy của A1 |

Các cải tiến kỹ thuật:

1. Gate một lớp: vẫn ghi fixed-binary Macro-F1 để báo cáo; perfect attack-only pool không bị coi là model sai vì macro=.5. Lỗi thật trên lớp đã quan sát vẫn được dùng. Thiếu benign support vẫn là unknown. Background update tiếp tục; emergency intensity chỉ thay khi có bằng chứng hiệu năng.
2. Tham số inactive bị loại khỏi danh sách differentiation và AdamW; kiểm tra parameter identity trước/sau, không suy từ prefix tên lớp. Log actual optimized/changed/representation/head fractions.
3. Đếm gradient training **và** conflict theo phase và số phần tử tensor được yêu cầu. Không đổi tên counter này thành FLOPs. Scope là post-warmup adaptation; warmup không nằm trong tổng đó.
4. Ghi ceiling thực chạy1.0. YAML cũ ghi2 nhưng callers dùng helper default1; giữ actual historical scale để không vô tình thay learning rate trong lúc sửa lỗi khác.
5. Ghi nhãn thực truy cập: bản original selection xem repair pool và gate, không log riêng128 training labels thành total supervision. Comparison chính phải gọi protocol-specific.
6. Bổ sung latency toàn chu kỳ, operational error counts, provenance code/data/config/environment và checkpoint chứa model/memory/RNG thật. Completed checkpoint không phải chức năng tự resume giữa stream.

Gate sửa có thể làm thay đổi kết quả. Cần báo negative findings nếu xảy ra; không chọn policy bằng một seed thắng. `legacy_fixed_macro` là đối chứng sensitivity, không là phiên bản mặc định mới. `original_review.enabled=false` chỉ dùng đối chiếu code cũ có giới hạn, không dùng làm phương pháp cải tiến.

## Thực nghiệm bổ sung cho vòng1

| Stage | Câu hỏi và điều kiện |
|---|---|
| routing | Native, random/fixed/MMD-only/conflict-only, all-module, head-only, k1/k2. Controls dùng cùng event schedule, epochs, replay budget, số mẫu và multiset plasticity của reference. All/head/k là sensitivity, không giả vờ cùng số optimized parameters |
| components | No replay, no weighting, no performance gate, no class balancing. Modular-all vs flat-all được match đúng tổng parameter count để kiểm tra backbone |
| gate | Gate sửa vs gate legacy; không khóa schedule vì cần đo thay đổi intensity |
| fairness | DASP cùng nền gốc với total query budget128 vs SSF128. Query trước labels, repair/gate disjoint. Chỉ thay supervision contract cần thiết; memory/backbone/background policy không thành A1 |
| permutation | Cùng feature order gốc và3hoán vị701/702/703; không thay ma trận chính |
| natural | NPZ chronological có timestamp/sourceIDs, global time order; chạy8methods. Các model/query seeds trên cùng capture không phải môi trường độc lập |

Event-matched routing là **offline diagnostic** lấy schedule từ reference; không gọi đó là deployable algorithm biết trước drift. Counterfactual models sẽ có predictions khác, nên native gate/endogenous schedules có thể khác nếu không khóa. Bản main không dùng schedule oracle.

Không cộng toàn bộ run thành independent replications. Paired keys là dataset/scenario/stream/model; bootstrap theo dataset-scenario và sensitivity theo dataset, Wilcoxon hai phía và Holm trong family. Báo coverage thiếu, không âm thầm drop pairs. Mean-window và cumulative recall/FNR/FPR được báo riêng. Legacy running-best window drop không phải fixed-task catastrophic forgetting.

8baseline names là thiết kế so sánh cũ. Các SSF/CDDA/RepShield implementations hiện có là adaptations; nhánh này không sửa lại baseline hay tự chứng nhận faithful native SOTA. Các lỗi baseline ngoài phạm vi DASP chưa được sửa ở đây. Do đó chưa được dùng kết quả để tuyên bố thắng năm native SOTA hoặc đạt Q1.

## Linux + NVIDIA GPU

Thư mục dữ liệu `prepared_streams_final` không nằm trong gói code. Copy nguyên NPZ và manifests cũ; không tái sinh ma trận chính bằng script mới. Ba tools preparation/validation/statistics bị thiếu được khôi phục từ Git ancestor `3c4224c9`; có provenance của source, chưa chứng minh bitwise equivalence với mọi NPZ lưu trữ.

Tạo virtualenv bằng Python3.12, cài CUDA-enabled PyTorch phù hợp driver theo [bộ chọn chính thức](https://pytorch.org/get-started/locally/), rồi dependencies dự án. Không ép cài Windows mkl/intel-openmp từ lockfile cũ lên Linux. Gói có file requirements Linux riêng; môi trường GPU phải được ghi `pip freeze` sau khi kiểm tra. Các pins này tách từ lock cũ, chưa phải môi trường Linux GPU đã xác nhận.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Cài PyTorch CUDA theo phiên bản/driver đã chọn, sau đó:
python -m pip install -r requirements-original-review-linux.txt
python -m pip install --no-deps -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m pytest tests/test_original_review.py tests/test_original_review_analysis.py -q
python tools/run_original_review.py --stage smoke --include-baselines --device cuda --output-root outputs_review_gpu_smoke --execute --checkpoints
python tools/analyze_original_review.py --stage smoke --output-root outputs_review_gpu_smoke
```

Xem kế hoạch chính, kiểm toàn bộ input trước khi train:

```bash
python tools/run_original_review.py --stage primary --device cuda --prepared-root /data/prepared_streams_final --output-root /results/dasp_review_primary
python tools/run_original_review.py --stage primary --device cuda --prepared-root /data/prepared_streams_final --output-root /results/dasp_review_primary --preflight
```

Chạy ma trận chính sau khi đã xem config/plan:

```bash
python tools/run_original_review.py --stage primary --device cuda --prepared-root /data/prepared_streams_final --output-root /results/dasp_review_primary --execute --resume
python tools/analyze_original_review.py --stage primary --output-root /results/dasp_review_primary
```

Chạy bộ bổ sung bằng cách thay `--stage primary` thành routing/components/gate/fairness/permutation. Mỗi bộ dùng output root riêng cho việc khóa kế hoạch và restart. Chạy pilot nhỏ bằng các flags dataset/scenario/seed rồi dùng root khác cho full stage; không thay kế hoạch của thư mục đã có completed runs.

Chronological:

```bash
python tools/prepare_revision_natural_stream.py --raw /data/RAW.csv --output /data/natural_capture.npz
python tools/run_original_review.py --stage natural --natural-stream capture_id=/data/natural_capture.npz --device cuda --output-root /results/dasp_review_natural --execute --resume
```

Chronological stream từ capture đã dùng để phát triển chỉ là kiểm tra setting khác, không tự thành untouched external validation. Chỉ khai báo unseen nếu có record split/tuning/source-ID disjointness chứng minh.

`--resume` chỉ reuse **run đã hoàn tất**, có cùng code/data/config/environment và output hashes. Không tiếp tục optimizer giữa một run đang dở; dùng output root mới khi interrupted run thiếu completion marker. Không tự overwrite archived outputs.

Validation cục bộ có CPU và CUDA trên Windows/RTX3050; Linux trên server chưa được chạy trực tiếp. Smoke là kiểm cơ chế, không là chứng minh predictive superiority. Đọc VALIDATION_ORIGINAL_REVIEW.md để biết kết quả thực chạy cuối.
