# Đánh giá cuối: chỉnh tham số trên dev, chạy một lần trên dữ liệu giữ lại

## Phương pháp đã chốt (`FINAL_METHOD` trong `tools/run_dasp_v2.py`)

- Module theo nhóm đặc trưng ngữ nghĩa (`model.feature_grouping = semantic`; nhóm volume / timing / tcp / protocol cho NetFlow v3 và CICFlowMeter, xem `src/dasp_nids/feature_groups.py`).
- Chẩn đoán drift có điều kiện theo lớp bằng nhãn giả (`dasp_v2.attribution = pseudo`): MMD theo nhóm đặc trưng trong từng lớp dự đoán, so với reference warmup. Không tốn nhãn; phân biệt được drift đo lường với thay đổi tỷ lệ lớp.
- Chẩn đoán điều khiển routing (`routing = attributed`, top-1): khi sửa chỉ cập nhật module của nhóm bị chẩn đoán, kèm head.
- Replay cân bằng lớp, logit adjustment theo prior của batch, hiệu chỉnh prior ACC.
- Chế độ cục bộ nghiêm ngặt (`final_strict_locality`): như trên, nhưng head chia khối. Khi sửa chỉ đổi module được chẩn đoán, khối head của nó và lớp cuối; cập nhật nền chỉ đổi lớp cuối.

## Phân vai dữ liệu

| Vai trò | Dataset | Seed | Kịch bản |
|---|---|---|---|
| Chỉnh tham số (dev) | NF-BoT-IoT-v3, NF-UNSW-NB15-v3 | 2001–2003, model 91 | S0–S7 (10), drift khu trú mức vừa FS/FR (8), label shift L1–L3 (3) |
| **Đánh giá cuối (giữ lại)** | NF-CSE-CIC-IDS2018-v3, NF-ToN-IoT-v3, CIC-IDS2017 | 3001–3005 (label shift 3001–3003), model 91–92 | S0–S7, FS/FR/XS/XR (16, gồm mức nghiêm trọng chưa dùng để chỉnh), L1–L3 |
| Đánh giá cuối, tự nhiên | NF-CSE, NF-ToN (stream chronological toàn capture) | 3001–3003, model 91–92 | Natural-Chronological |
| Phụ: dataset dev với seed mới | NF-BoT (3001–3005), NF-UNSW (3001–3002) | model 91–92 | S0–S7, báo cáo tách riêng |

Không quyết định thiết kế hay tham số nào dùng dữ liệu giữ lại. Trên dữ liệu giữ lại chỉ chạy preflight đầu vào và kiểm tra mô hình dựng được với bộ đặc trưng CIC; chưa chạy phương pháp nào.

## Lệnh chạy trên server (CPU, Python 3.12, PyTorch 2.3.1 CPU)

Chuẩn bị (một lần): push nhánh, clone, cài môi trường, chuyển `prepared_streams_v2` rồi kiểm tra bằng `sha256sum -c SHA256SUMS --quiet`. Xem hướng dẫn trước.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; N=19; mkdir -p results/logs
```

**Bước 1: chỉnh tham số trên dev** (21.168 run: 7 phương pháp × 24 cấu hình × 126 cell; khoảng 60–90 giờ-CPU, tức khoảng 3,5–5 giờ với 19 nhân):

```bash
python tools/run_dasp_v2.py --stage dev --prepared-root prepared_streams_v2 --output-root results/dev --preflight
```

```bash
for k in $(seq 1 $N); do nohup python tools/run_dasp_v2.py --stage dev --prepared-root prepared_streams_v2 --output-root results/dev --execute --resume --shard $k/$N > results/logs/dev_$k.log 2>&1 & done
```

```bash
python tools/analyze_dasp_v2.py --stage dev --output-root results/dev
```

Lệnh phân tích ghi `results/dev/dev_selection.json`: với mỗi phương pháp, cấu hình có Macro-F1 trung bình cao nhất trên 126 cell dev; hòa trong phạm vi 0,002 thì chọn cấu hình ít lượt gradient hơn.

**Bước 2: đánh giá cuối** (14.790 run: 986 cell × 15 điều kiện, gồm phương pháp cuối, static, 6 baseline đã chỉnh, chế độ cục bộ nghiêm ngặt và 6 ablation):

```bash
python tools/run_dasp_v2.py --stage final --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/final --preflight
```

```bash
for k in $(seq 1 $N); do nohup python tools/run_dasp_v2.py --stage final --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/final --execute --resume --shard $k/$N > results/logs/final_$k.log 2>&1 & done
```

```bash
python tools/analyze_dasp_v2.py --stage final --output-root results/final
```

Theo dõi tiến độ bằng `tail -qn1 results/logs/final_*.log`. Bị ngắt thì chạy lại đúng lệnh với **cùng N**; `--resume` bỏ qua các run đã xong và kiểm tra hash.

## Đầu ra của bước 2

- `final_summary.csv`: trung bình theo (giữ lại / dev-seed-mới) × loại kịch bản × điều kiện. Gồm Macro-F1 trung bình cửa sổ và tích lũy, attack recall, FPR benign, F1 sau thay đổi và khi concept quay lại, chi phí gradient.
- `final_by_dataset.csv`: Macro-F1 theo từng dataset.
- `final_analysis.json`: so sánh cặp ở mức cell (trung bình theo seed); bootstrap phân cấp (dataset rồi kịch bản), Wilcoxon hai phía, Holm riêng cho họ baseline và họ ablation, tách theo từng loại kịch bản. Stream tự nhiên dùng Macro-F1 tích lũy. Thêm độ chính xác chẩn đoán top-1 lúc drift bắt đầu, theo dataset và nhóm đặc trưng, và tỷ lệ tham số thay đổi mỗi cửa sổ của hai chế độ.
- `final_runs.csv`: từng run.

Gửi lại các file này cùng `results/dev/dev_selection.json` và `pip freeze` của server để viết phần kết quả.
