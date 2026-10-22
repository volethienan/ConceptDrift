# E4, E6, E15: shortcut ablation, label budget/delay/query policy sweep, Bayesian ROPE

Ba thí nghiệm P0 của báo cáo Q1 (`reports/NIDS thích ứng trôi dạt.md`). Cả ba dùng **cấu hình đã chọn trên dev** (`dev_selection.json`), không tune lại. Mọi dataset giờ là dữ liệu phát triển, vì kết quả held-out đã được xem.

## Thay đổi trong code

| Thành phần | Nơi | Ý nghĩa |
|---|---|---|
| `baselines.query_policy` | `config.py`, `runner.QUERY_POLICIES` | `uniform` (hợp đồng cũ, giữ nguyên byte), `uncertainty` (gần ngưỡng quyết định nhất), `mixed` (một nửa uncertainty, một nửa uniform), `pred_stratified` (một nửa dự đoán benign, một nửa dự đoán attack). Không chính sách nào đọc nhãn: chỉ dùng attack score mà chính phương pháp tính lúc dự đoán window. |
| `StreamingExperiment._query_indices` | `runner.py` | Một hàm truy vấn chung cho baseline ngang ngân sách, DASP-v2 và 4 adapter SOTA. Adapter dùng ngưỡng riêng của nó (`adapter.threshold`); AOC-IDS chỉ trả nhãn 0/1 nên `uncertainty` của nó thực chất là uniform. |
| `data.drop_features` | `config.py`, `data.load_prepared_stream` | Bỏ cột theo tên khi nạp stream. Nhóm ngữ nghĩa vẫn kiểm tra schema chặt: một feature thiếu mà không khai báo bỏ thì vẫn báo lỗi. |
| Stage `sweep`, `shortcut` | `tools/run_dasp_v2.py` | Điều kiện có dạng `<phương pháp>@<thiết lập>`. |
| `bayesian_signed_rank`, stage `rope`, `sweep`, `shortcut` | `tools/analyze_dasp_v2.py` | Mọi contrast có thêm `bayes_signed_rank` (Benavoli et al. 2017; ROPE ±0,01 Macro-F1, đổi bằng `--rope`). |

Test: `tests/test_query_shortcut_rope.py`. Có một test lật mọi nhãn nằm ngoài truy vấn của từng chính sách và kiểm tra score lúc dự đoán không đổi, với ER, DASP-v2 và ARF.

Sửa kèm: `tools/analyze_dasp_v2.py` (bản chưa commit) bị `IndentationError` ở dòng `CHANGE_POINTS['CX-…']`, nên mọi lệnh phân tích đều hỏng. Đã sửa.

## E15: đã chạy trên snapshot final, không cần server

```bash
python tools/analyze_dasp_v2.py --stage rope --runs results_snapshot/final/final_runs.csv.gz --output-root results_snapshot/final
```

Kết quả nằm ở `results_snapshot/final/rope_analysis.json` và `rope_table.csv`. Trên held-out, khi so với DASP-v2 final:

| Track | ER: P(tốt hơn) / P(tương đương) | ARF: P(tương đương) | Ghi chú |
|---|---|---|---|
| attack_family | 0.23 / 0.77 | 0.99 | ER hơn có ý nghĩa (Holm p 7e-7) nhưng phần lớn trong ROPE |
| feature_shift moderate / severe | 0.00 / 1.00 | 1.00 | |
| label_shift | 0.01 / 0.99 | 0.99 | |
| natural | 0.18 / 0.65 | 0.85 | chỉ 2 cell: posterior còn rất bất định |

AOC/CND/SOUL/static: P(tệ hơn) ≈ 1 trên các track controlled.

Cách viết đúng trong bài: ER, ARF, SSF và DASP-v2 tạo thành một nhóm đầu **tương đương thực tế** (ROPE ±0,01). Khoảng cách của ER có ý nghĩa thống kê nhưng nhỏ hơn mức có ý nghĩa vận hành.

## E6 và E4: chạy trên server

Code trong worktree chưa commit. Phải commit và push nhánh `dasp-v2-infra` trước, rồi trên server:

```bash
cd /workspace/anvlt/ConceptDrift && source .venv/bin/activate && git pull && python -c "import numpy, river; print(numpy.__version__, river.__version__)"
```

Dòng cuối phải in `1.26.4 0.23.0`.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; N=19; mkdir -p results/logs
```

Shard theo cell. Sweep và shortcut đều có 38 cell (30 controlled + 8 chronological), nên `N=19` chia đều 2 cell mỗi shard.

### E6: sweep (4.978 run)

26 thiết lập cho mỗi phương pháp: ngân sách {32, 64, 128, 256, 512} × 4 chính sách ở độ trễ 1, cộng độ trễ {0, 1, 5, 10} × {uniform, mixed} ở ngân sách 128. Phương pháp: DASP-v2 final, ER, Online MLP, SSF-budget, ARF, cộng static một lần mỗi cell. Cell: 5 dataset × {S1-Sudden, S2-Gradual, S6-Combined} × 2 seed, cộng 4 stream chronological × 2 seed.

```bash
python tools/run_dasp_v2.py --stage sweep --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/sweep --seeds-per-dataset 2 --preflight
```

```bash
for k in $(seq 1 $N); do nohup python tools/run_dasp_v2.py --stage sweep --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/sweep --seeds-per-dataset 2 --execute --resume --shard $k/$N > results/logs/sweep_$k.log 2>&1 & done
```

```bash
python tools/analyze_dasp_v2.py --stage sweep --output-root results/sweep
```

Đầu ra:
- `sweep_summary.csv`: điểm (Macro-F1; stream chronological dùng cumulative), FPR, recall, chi phí theo track × phương pháp × ngân sách × độ trễ × chính sách.
- `sweep_analysis.json`, gồm hai họ contrast:
  - `<đối thủ>_vs_final_per_setting`: thứ hạng ER/ARF/MLP/SSF so với DASP-v2 có giữ ở mọi thiết lập không.
  - `<phương pháp>_policy_vs_uniform`: chính sách nào tiêu ngân sách tốt hơn uniform ở cùng ngân sách.

### E4: shortcut ablation (876 run)

Bộ bỏ feature: `none`, `ports` (L4_SRC_PORT, L4_DST_PORT; CIC-IDS2017: Destination Port), `ttl` (MIN_TTL, MAX_TTL; CIC-IDS2017 không có nên bỏ qua), `ports_ttl`. Phương pháp: DASP-v2 final, ER, Online MLP, SSF-budget, ARF, static. Cell: 5 dataset × {S0, S1-Sudden, S6-Combined} × 2 seed, cộng 4 stream chronological × 2 seed.

```bash
python tools/run_dasp_v2.py --stage shortcut --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/shortcut --seeds-per-dataset 2 --preflight
```

```bash
for k in $(seq 1 $N); do nohup python tools/run_dasp_v2.py --stage shortcut --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/shortcut --seeds-per-dataset 2 --execute --resume --shard $k/$N > results/logs/shortcut_$k.log 2>&1 & done
```

```bash
python tools/analyze_dasp_v2.py --stage shortcut --output-root results/shortcut
```

Đầu ra:
- `shortcut_summary.csv`: điểm theo track × dataset × phương pháp × bộ bỏ.
- `shortcut_analysis.json`, gồm hai họ contrast:
  - `<phương pháp>_drop_vs_none`: mức tụt khi bỏ shortcut, chỉ trên cell mà bộ đó được chạy.
  - `<đối thủ>_vs_final_per_drop`: thứ hạng có giữ khi bỏ shortcut không.

Gửi lại `results/sweep/{sweep_summary.csv,sweep_analysis.json}` và `results/shortcut/{shortcut_summary.csv,shortcut_analysis.json}` để viết phần kết quả.
