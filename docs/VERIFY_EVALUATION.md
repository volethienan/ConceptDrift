# Xác minh trong cùng window khi nhãn đến trễ (stage `verify`)

## Bài toán

Trên stream natural, attack đến theo đợt dài 1–3 window. Recall của một đợt chỉ lên khi đã có nhãn của loại attack đó. Vì vậy với độ trễ d, những window đầu của đợt bị bỏ sót gần hết. Ví dụ CSE ở d5: ER có recall 0 trên mọi đợt.

Nhãn bulk vẫn đến trễ d window. Nhưng analyst có thể xác minh ngay **vài flow của window hiện tại** (fast lane). Vì flow trong một đợt rất đồng nhất, một nhãn đã xác minh có thể đại diện cho cả cụm.

## Hợp đồng nhãn

- Mỗi window có tổng cộng `update_label_budget` = 128 nhãn:
  - `fast_label_budget` nhãn được lộ ngay sau khi dự đoán window;
  - phần còn lại đến trễ `label_delay` window, chọn uniform trong các flow fast lane chưa hỏi.
- Fast lane chỉ đọc nhãn qua `verified_propagation.LabelOracle`. Oracle ghi lại từng truy vấn và chặn khi vượt ngân sách.
- Flow bị fast lane cảnh báo được OR vào dự đoán của chính window đó, giống phân loại theo incident trong SOC.
- Mô hình học ngay trên các nhãn nhanh đã xác minh. Huấn luyện thêm bằng các flow bị cảnh báo (làm nhãn attack giả) là tuỳ chọn `fast_lane_pseudo_labels`, mặc định tắt; trên dev nó đẩy FPR của ToN từ 5% lên 7%.
- Nếu fast lane không dùng hết ngân sách, phần thừa thành nhãn nhanh uniform.

Test: `tests/test_fast_lane.py` lật mọi nhãn nằm ngoài truy vấn fast và bulk, rồi kiểm tra dự đoán đã hợp nhất không đổi, cho cả 5 kiểu fast lane. `tests/test_verified_propagation.py` kiểm tra điều tương tự ở mức thuật toán.

## Các kiểu fast lane (`baselines.fast_lane`)

| Kiểu | Cơ chế |
|---|---|
| `uniform` | Nhãn nhanh uniform, không lan truyền. Đây là baseline fast lane. |
| `representative` | Chia flow mô hình đoán benign thành k cụm k-means. Hỏi 1 nhãn mỗi cụm (flow gần tâm), lan truyền không kiểm định. Đây là pilot "cảnh báo rồi xác minh". |
| `purity` | Sàng mỗi cụm bằng 1 nhãn ngẫu nhiên. Cụm ra attack được rút tiếp đến đủ r nhãn ngẫu nhiên; nếu cả r đều attack và cận dưới Clopper-Pearson ≥ 1−ε thì cảnh báo cả cụm. Cụm lẫn nhãn thì tách đôi. Cụm có tỉ lệ benign > ε lọt qua với xác suất ≤ (1−ε)^r. |
| `uniform_knn` | Nhãn nhanh uniform; mỗi flow mô hình đoán benign nhận nhãn của flow đã xác minh gần nhất. |
| `hierarchical` | Dasgupta & Hsu (ICML 2008) trên cây Ward. |

## Điều kiện (26 mỗi cell)

| Nhóm | Điều kiện | Ghi chú |
|---|---|---|
| Không fast lane | `er@d{1,5}` | ER, 128 nhãn trễ d window |
| Baseline độ trễ nhãn | `er_iwms@d{1,5}` | IWMS (Csaba et al., NeurIPS 2024): replay mẫu nhớ gần nhất với dữ liệu mới chưa nhãn, `baselines.replay_sampling=iwms` |
| Baseline active learning | `er_uncertainty@d{1,5}` | Hỏi 128 nhãn gần ngưỡng nhất (kiểu Olímpio et al. 2023) |
| Baseline stream | `arf@d{1,5}` | ARF, cấu hình đã chọn trên dev |
| Fast lane, k = 16 | `er+{uniform,representative,purity,uniform_knn,hierarchical}@d{1,5}_k16` | Method chính là `purity` |
| Ablation nhãn giả | `er+purity_pseudo@d{1,5}_k16` | Huấn luyện thêm bằng flow bị cảnh báo |
| Ablation ngân sách nhanh | `er+{uniform,purity}@d5_k{4,8,32}` | |

**Dataset dev:** NF-UNSW, NF-CSE, NF-ToN; mỗi dataset 3 stream seed × 3 model seed (91–93), tổng 702 run. NF-BoT bị loại: 64% flow benign có bản trùng y hệt mang nhãn attack, nên không detector flow nào có FPR benign dưới 61% (`label_twins.py`, xem HANDOFF).

**Held-out:** CIC-IDS2017 natural, 3 stream seed × 3 model seed, tổng 234 run, chỉ lên kế hoạch khi có `--heldout`.

## Lệnh trên server

```bash
cd /workspace/anvlt/ConceptDrift && source .venv/bin/activate && git pull && python -c "import numpy, river; print(numpy.__version__, river.__version__)"
```

Phải in ra `1.26.4 0.23.0`. Stream CIC-IDS2017 natural không có trong `SHA256SUMS`, nên phải chép thư mục `prepared_streams_v2/natural/cic-ids2017/` lên server, hoặc dựng lại bằng `tools/build_natural_cic_stream.py --stream-seeds 4001,4002,4003`.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; N=27; mkdir -p results/logs
```

```bash
for k in $(seq 1 $N); do nohup python tools/run_dasp_v2.py --stage verify --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/verify --execute --resume --shard $k/$N > results/logs/verify_$k.log 2>&1 & done
```

```bash
python tools/analyze_dasp_v2.py --stage verify --output-root results/verify
```

Gửi lại `results/verify/{verify_summary.csv,verify_analysis.json,verify_runs.csv}`. **Không** chạy `--heldout` cho đến khi thiết kế đã chốt trên dev.
