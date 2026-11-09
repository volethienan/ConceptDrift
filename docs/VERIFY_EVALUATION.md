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

## Lệnh

```bash
python tools/run_dasp_v2.py --stage verify --selection results/dev/dev_selection.json --prepared-root prepared_streams_v2 --output-root results/verify --seeds-per-dataset 3 --execute --resume --shard $k/$N
```

Dev gồm 4 dataset natural × 3 seed × 24 điều kiện, tổng 288 run:
- ER ở d1 và d5;
- 5 kiểu fast lane × k ∈ {8, 16} × d ∈ {1, 5};
- purity có nhãn giả (k = 16) ở d1 và d5.

**Held-out** là stream natural CIC-IDS2017, seed 4001–4003, dựng bằng `tools/build_natural_cic_stream.py`. Chỉ lên kế hoạch khi có `--heldout`, và **chỉ chạy một lần, sau khi thiết kế đã chốt trên dev**.
