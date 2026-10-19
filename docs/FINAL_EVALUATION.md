# Đánh giá cuối: chỉnh tham số trên dev, chạy một lần trên dữ liệu giữ lại

## Phương pháp đã chốt (`FINAL_METHOD` trong `tools/run_dasp_v2.py`)

- Module theo nhóm đặc trưng ngữ nghĩa (`model.feature_grouping = semantic`; nhóm volume / timing / tcp / protocol cho NetFlow v3 và CICFlowMeter, xem `src/dasp_nids/feature_groups.py`).
- Chẩn đoán drift có điều kiện theo lớp bằng nhãn giả (`dasp_v2.attribution = pseudo`): MMD theo nhóm đặc trưng trong từng lớp dự đoán, so với reference warmup. Không tốn nhãn; phân biệt được drift đo lường với thay đổi tỷ lệ lớp.
- Chẩn đoán điều khiển routing (`routing = attributed`, top-1): khi sửa chỉ cập nhật module của nhóm bị chẩn đoán, kèm head.
- **Hiệu chỉnh theo routing** (`dasp_v2.realign = attributed`, `FINAL_REALIGN`). Ngay lúc dự đoán, không cần nhãn và không tốn gradient. Routing quyết định *có* can thiệp không và can thiệp *ở đâu*:
  1. Nhóm bị chẩn đoán phải vượt `realign_z` ở 2 cửa sổ liên tiếp, hoặc vượt gấp 5 lần ngưỡng.
  2. Ước lượng ánh xạ affine trung vị/IQR theo lớp, dùng nhãn giả của mô hình đã che module bị drift. Chỉ đổi những đặc trưng lệch quá 2 lần nhiễu warmup.
  3. Chỉ chấp nhận khi drift của nhóm biến mất sau hiệu chỉnh, tức đó là lỗi đo đạc chứ không phải họ tấn công mới.
  4. Ánh xạ được ước lượng lại mỗi cửa sổ, tự gỡ khi dữ liệu gốc trở lại bình thường, và bị gỡ khi nhãn trễ cho thấy loss tăng.
  Ngưỡng của bước này cố định trên dev; lưới dev chỉnh phần huấn luyện khi đã bật hiệu chỉnh.
- Replay cân bằng lớp, logit adjustment theo prior của batch, hiệu chỉnh prior ACC.
- Chế độ cục bộ nghiêm ngặt (`final_strict_locality`): như trên, nhưng head chia khối. Khi sửa chỉ đổi module được chẩn đoán, khối head của nó và lớp cuối; cập nhật nền chỉ đổi lớp cuối.

## Đối chứng

| Nhóm | Phương pháp | Bộ (`--suite`) |
|---|---|---|
| Học liên tục kinh điển | Online MLP, Experience Replay, A-GEM, Online EWC (Fisher thực nghiệm) | `core` |
| NIDS / drift | SSF (budget), CDDA-MD v2 | `core` |
| NIDS / drift, chuyển thể từ mã tác giả | AOC-IDS-adapted (INFOCOM 2024), CND-IDS-adapted (DAC 2025), SOUL-adapted (2024), ARF-adapted (River 0.23.0) | `sota` |

Mọi phương pháp, kể cả bốn bản chuyển thể, chạy cùng một hợp đồng: warmup có nhãn đầy đủ, sau đó mỗi cửa sổ nhận nhãn của đúng 128 mẫu truy vấn đồng đều không nhìn nhãn, trễ một cửa sổ. Mỗi phương pháp có 24 cấu hình, chọn trên dev. Mã chuyển thể nằm ở `src/dasp_nids/sota_adapters.py`, ghi rõ nguồn (commit) và từng chỗ khác bản gốc:

- **Không có ranh giới task**: mỗi cửa sổ là một task. Chỉ 128 mẫu truy vấn có nhãn; phần còn lại được dùng như dữ liệu không nhãn khi phương pháp gốc có dùng (AOC, CND, SOUL đều dùng nhãn giả).
- **Bộ nhớ có giới hạn** bằng `dasp.memory_size`, giống replay của các phương pháp khác. Bản gốc AOC giữ toàn bộ lịch sử.
- **Ngưỡng quyết định** mà bản gốc chọn trên dữ liệu kiểm thử hoặc validation có nhãn thì ở đây chọn trên nhãn truy vấn đã nhận, tối đa Macro-F1. Riêng CND có thêm lựa chọn ngưỡng phân vị trên dữ liệu bình thường.
- **Chuẩn hóa đầu vào**: ba mạng nơ-ron gốc dùng min–max. Ở đây min–max fit trên warmup có nhãn, cắt về [0, 1]. ARF dùng cây nên không chuẩn hóa.
- **Bổ sung**: CND gom cụm bằng k-means khởi tạo ngẫu nhiên, số cụm giới hạn ở 1/8 cửa sổ. SOUL lấy gradient qua loss chưng cất trên dữ liệu không nhãn, trong khi mã gốc tách gradient của số hạng này nên số hạng đó không có tác dụng.
- Mỗi run ghi `total_ground_truth_update_labels`, `warmup_labels` và `total_pseudo_labels_used`. Test `tests/test_sota_adapters.py` lật mọi nhãn nằm ngoài tập truy vấn và kiểm tra dự đoán không đổi.

## Phân vai dữ liệu

| Vai trò | Dataset | Seed | Kịch bản |
|---|---|---|---|
| Chỉnh tham số (dev) | NF-BoT-IoT-v3, NF-UNSW-NB15-v3 | 2001–2003, model 91 | S0–S7 (10), drift khu trú mức vừa FS/FR (8), label shift L1–L3 (3) |
| **Đánh giá cuối (giữ lại)** | NF-CSE-CIC-IDS2018-v3, NF-ToN-IoT-v3, CIC-IDS2017 | 3001–3005 (label shift 3001–3003), model 91–92 | S0–S7, FS/FR/XS/XR (16, gồm mức nghiêm trọng chưa dùng để chỉnh), L1–L3 |
| Đánh giá cuối, tự nhiên | NF-CSE, NF-ToN (stream chronological toàn capture) | 3001–3003, model 91–92 | Natural-Chronological |
| Phụ: dataset dev với seed mới | NF-BoT (3001–3005), NF-UNSW (3001–3002) | model 91–92 | S0–S7, báo cáo tách riêng |

Không quyết định thiết kế hay tham số nào dùng dữ liệu giữ lại.

## Lệnh chạy trên server (CPU, Python 3.12, PyTorch 2.3.1 CPU, 18 nhân)

Repo: `/workspace/anvlt/ConceptDrift`, venv `.venv`, dữ liệu `prepared_streams_v2` (đã kiểm tra `sha256sum -c SHA256SUMS --quiet`). Mọi phương pháp chạy lại từ đầu bằng code hiện tại.

### Bước 0: dừng lần chạy cũ, cất kết quả cũ, cập nhật code

```bash
cd /workspace/anvlt/ConceptDrift && source .venv/bin/activate
```

```bash
pkill -f "tools/run_dasp_v2.py"; mkdir -p results_old && mv results/dev results_old/dev_before_realign 2>/dev/null; mkdir -p results/logs
```

```bash
git pull && pip install river==0.23.0 && python -c "import numpy, river; print(numpy.__version__, river.__version__)"
```

Dòng cuối phải in `1.26.4 0.23.0`. River 0.25 trở lên đòi numpy 2, nên không dùng.

```bash
python tools/run_dasp_v2.py --stage smoke --output-root /tmp/dasp_smoke --execute
```

### Bước 1: chỉnh tham số trên dev (33.264 run: 11 phương pháp × 24 cấu hình × 126 cell)

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; N=18
```

```bash
python tools/run_dasp_v2.py --stage dev --prepared-root prepared_streams_v2 --output-root results/dev --preflight
```

```bash
for k in $(seq 1 $N); do nohup python tools/run_dasp_v2.py --stage dev --prepared-root prepared_streams_v2 --output-root results/dev --execute --resume --shard $k/$N > results/logs/dev_$k.log 2>&1 & done
```

Ước lượng khoảng 16 giờ với 18 nhân; phần lớn thời gian là CND-IDS và ARF. Theo dõi tiến độ bằng `tail -qn1 results/logs/dev_*.log`.

```bash
python tools/analyze_dasp_v2.py --stage dev --output-root results/dev
```

Lệnh này ghi `results/dev/dev_selection.json`: với mỗi phương pháp, cấu hình có Macro-F1 trung bình cao nhất trên 126 cell dev; hòa trong phạm vi 0,002 thì chọn cấu hình ít lượt gradient hơn.

### Bước 2: đánh giá cuối (22.678 run: 986 cell × 23 điều kiện)

Gồm phương pháp cuối, static, 6 baseline, 4 bản chuyển thể SOTA, chế độ cục bộ nghiêm ngặt, 6 ablation và 4 ablation hiệu chỉnh: không hiệu chỉnh, nhóm ngẫu nhiên, mọi nhóm mọi cửa sổ, và bỏ kiểm định.

```bash
python tools/run_dasp_v2.py --stage final --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/final --preflight
```

```bash
for k in $(seq 1 $N); do nohup python tools/run_dasp_v2.py --stage final --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/final --execute --resume --shard $k/$N > results/logs/final_$k.log 2>&1 & done
```

```bash
python tools/analyze_dasp_v2.py --stage final --output-root results/final
```

Nếu bị ngắt, chạy lại đúng lệnh với **cùng N**; `--resume` sẽ bỏ qua các run đã xong và kiểm tra hash. Trong lúc đang chạy, không `git pull`: mỗi run ghi hash mã nguồn.

## Đầu ra của bước 2

- `final_summary.csv`: trung bình theo (giữ lại / dev-seed-mới) × loại kịch bản × điều kiện. Gồm Macro-F1 trung bình cửa sổ và tích lũy, attack recall, FPR benign, F1 sau thay đổi và khi concept quay lại, chi phí gradient.
- `final_by_dataset.csv`: Macro-F1 theo từng dataset.
- `final_analysis.json`: so sánh cặp ở mức cell (trung bình theo seed); bootstrap phân cấp (dataset rồi kịch bản), Wilcoxon hai phía, Holm riêng cho họ baseline (10 đối chứng cùng static) và họ ablation, tách theo từng loại kịch bản. Stream tự nhiên dùng Macro-F1 tích lũy. Thêm độ chính xác chẩn đoán top-1 lúc drift bắt đầu, theo dataset và nhóm đặc trưng, và tỷ lệ tham số thay đổi mỗi cửa sổ của hai chế độ. Mục `realignment` cho biết lần hiệu chỉnh đầu có trúng nhóm bị drift không, và số lần hiệu chỉnh ở những kịch bản không có lỗi đo đạc.
- `final_runs.csv`: từng run.

Gửi lại các file này cùng `results/dev/dev_selection.json` và `pip freeze` của server để viết phần kết quả.
