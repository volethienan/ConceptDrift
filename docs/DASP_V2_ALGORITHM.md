# DASP-v2 (`modular_dasp_v2`): thuật toán đã cài đặt

Phiên bản `modular-dasp-v2-20261018-r0`. Code ở [`src/dasp_nids/dasp_v2.py`](../src/dasp_nids/dasp_v2.py), cấu hình ở `DaspV2Config` (`config.py`, khóa `dasp_v2`). Tương ứng mục 4 của [PREREGISTRATION_DASP_V2.md](PREREGISTRATION_DASP_V2.md). **Chưa chạy trên dữ liệu thật; chưa có cấu hình được chọn.**

## Một lượt cập nhật (khi nhãn của cửa sổ t đến, trễ một cửa sổ)

1. **Hợp đồng nhãn.** Truy vấn đúng `update_label_budget` (128) chỉ số, đồng nhất và không nhìn nhãn, cùng seed với mọi baseline dùng ngân sách. Chia 75% để train, 25% (32) cho gate. Không đọc nhãn nào khác của cửa sổ.
2. **C1, gate dựa trên bằng chứng.** Pool gate gồm 32 nhãn gate hiện tại và nhãn gate của `gate_buffer_windows` cửa sổ trước (mặc định 2, tức tối đa 96). Một lớp chỉ được đánh giá khi có ít nhất `gate_min_support` nhãn (mặc định 20). Mô hình được coi là suy giảm nếu với một lớp được đánh giá, cận dưới Wilson của recall nhỏ hơn `gate_threshold` (0,80, z = 1,96). Nói cách khác: sửa khi **không chứng nhận được** mô hình còn tốt. Không lớp nào đủ mẫu thì trạng thái là `insufficient_support`: không bao giờ được đọc thành "tốt" hay "suy giảm".
3. **Quyết định.** Sửa (repair) khi MMD báo drift **hoặc** gate báo suy giảm. Ngược lại là cập nhật nền.
4. **C2, routing theo lợi ích đo được** (chỉ dùng khi sửa). Trên bản sao mô hình, chạy `utility_probe_epochs` bước train cho (a) chỉ head và (b) từng module cộng head. Lợi ích của module = loss của (a) trừ loss của (b), đo bằng cross-entropy cân bằng lớp trên pool gate. Chọn tối đa `routing_top_k` module có lợi ích lớn hơn `utility_min_gain`. Không module nào có lợi thì chỉ sửa head. Mọi bước thử được tính vào pha gradient `routing_probe`.
5. **Cập nhật nền.** `classifier` (mặc định): chỉ cập nhật head. `native`: như DASP gốc, top-k module theo conflict với plasticity 0,25. `none`: không train.
6. **C3, logit adjustment** (Menon et al., ICLR 2021): khi train cộng τ·log π_y vào logit. π ước lượng từ nhãn đã đến trong bộ nhớ (làm trơn Laplace). Dự đoán dùng logit gốc.
7. **Train.** Replay từ bộ nhớ cộng trọng số mẫu mới (`new_sample_weight` 3). Module không được chọn bị loại khỏi cả optimizer lẫn autograd, và được kiểm tra không đổi sau khi train (`_locality` báo lỗi nếu có thay đổi).
8. **C4, head nhỏ** là thiết lập kiến trúc: `model.modular_head_dim` (None = head cũ, 0 = head tuyến tính, k = lớp fusion rộng k). Tỷ lệ tham số của head với 49 đặc trưng: head cũ 81%, k=16 52%, k=8 35%, tuyến tính khoảng 12%.

## Đối chứng và ablation có sẵn qua cấu hình

| Thành phần | Công tắc |
|---|---|
| C1 | `gate_policy`: `wilson` / `point` / `off` |
| C2 | `routing`: `utility` / `native` (MMD + conflict như DASP gốc) / `all` / `classifier` |
| C3 | `logit_adjustment_tau`: 0 tắt |
| C4 | `model.modular_head_dim` |
| Replay | `replay_enabled` |
| Cập nhật nền | `background` |

## Ghi log mỗi cửa sổ

Gồm: query indices; trạng thái gate, số mẫu và cận Wilson của từng lớp; lợi ích routing của từng module; module được sửa; head có được cập nhật không; vector logit adjustment; số gradient theo pha (training, routing_probe, conflict) và số phần tử gradient được yêu cầu (không phải FLOPs); tỷ lệ tham số trong optimizer; số tham số thay đổi; thời gian toàn chu kỳ. Summary ghi thêm số repair, phân bố trạng thái gate, FPR benign tích lũy và trung bình theo cửa sổ.

## Kiểm chứng

`tests/test_dasp_v2.py`, 9 test: cận Wilson khớp giá trị tham chiếu; gate không coi thiếu mẫu là bằng chứng; Wilson chặt hơn ước lượng điểm; prior của logit adjustment; head tuyến tính và mặc định không đổi khởi tạo; đổi nhãn ngoài query không làm đổi mô hình; routing đóng băng module không chọn và tính chi phí thử; cập nhật nền chỉ đổi head; chạy end-to-end đúng ngân sách 128 nhãn.

## Đề xuất lưới phát triển (cần duyệt trước pha dev, tối đa 24 cấu hình)

| Siêu tham số | Giá trị |
|---|---|
| `model.modular_head_dim` (C4) | 0, 16 |
| `gate_min_support` × `gate_threshold` (C1, Wilson) | (20; 0,80), (20; 0,70), (40; 0,80) |
| `routing_top_k` (C2) | 1, 2 |
| `logit_adjustment_tau` (C3) | 0, 1 |

2 × 3 × 2 × 2 = 24 cấu hình. Cố định: `gate_buffer_windows` 2, `utility_probe_epochs` 1, `background` classifier, `repair_epochs` 2. Quy tắc chọn: mean-window Macro-F1 trung bình trên 60 cell dev (2 dataset × 10 kịch bản × 3 seed, model seed 91). Hòa trong phạm vi 0,002 thì chọn cấu hình có ít gradient element hơn.
