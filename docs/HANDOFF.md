# HANDOFF: trạng thái dự án DASP-v2 (cập nhật 2026-10-23)

Đọc file này trước khi làm gì. Nó thay cho bộ nhớ của các phiên Claude trước.

## Bối cảnh

- Bài DASP-NIDS bị IEEE Access reject (2 vòng). DASP-v2 là **bài mới** nộp tạp chí Q1 khác, không phải revision.
- Người dùng muốn **lặp nhanh**, không làm thủ tục khóa, hash hay pre-registration.
- **Thử cục bộ trước rồi mới đề xuất chạy server.** Người dùng đã phản đối việc đề xuất server khi chưa có kết quả mock.
- Trả lời **ngắn gọn**, bằng tiếng Việt.

## Vị trí

| Thứ | Đường dẫn |
|---|---|
| Code DASP-v2 | worktree `.worktrees/dasp-v2`, nhánh `dasp-v2-infra` (LODO, compound, E4/E6/E15 đã push). Chưa commit: fast lane/xác minh cùng window (`verified_propagation.py`, stage `verify`), builder CIC natural. |
| Checkout chính | Có sửa đổi chưa commit của người dùng; **không đụng vào**. |
| Dữ liệu | `prepared_streams_v2/` ở checkout chính (42 GB: controlled, `feature_shift/`, `label_shift/`, `natural/`) |
| Python | `.venv-revision/Scripts/python.exe` (Py 3.12, torch 2.3.1 CPU). Chạy test với `PYTHONPATH="src;."`. |
| Server | Khoảng 18–25 nhân, repo `/workspace/anvlt/ConceptDrift`. Pin `river==0.23.0` (numpy 1.26.4). |
| Kết quả final (held-out) | `results_snapshot/final/` (22.678 run). Thêm `rope_analysis.json` và `rope_table.csv` (E15). |
| Pilot cục bộ E4/E6 | `results_snapshot/pilot_local_20261022/` |
| Báo cáo đánh giá Q1 + tài liệu | Checkout chính: `reports/NIDS thích ứng trôi dạt.md`, `research_notes/NIDS thích ứng trôi dạt/` |
| Lệnh E4/E6/E15 | `docs/E4_E6_E15_EVALUATION.md` |
| Hướng mới: xác minh cùng window | `docs/VERIFY_EVALUATION.md`; held-out niêm phong `prepared_streams_v2/natural/cic-ids2017` (seed 4001–4003, chưa chạy method nào) |

## Hướng mới (2026-10-23): xác minh trong cùng window khi nhãn trễ

Người dùng muốn bài **method mới** (một nguyên lý mới, không ghép module), **không còn làm concept drift** (đã bão hoà). Chủ đề mới: phát hiện xâm nhập khi nhãn đến trễ.

- **Recall của một đợt attack chỉ phụ thuộc vào lúc nhãn đầu tiên của nó đến.** Đợt dài 1–3 window, nên ở d5 ER bỏ sót gần hết.
- **Cơ chế (`baselines.fast_lane=purity`):** trong 128 nhãn mỗi window, 16 nhãn được xác minh ngay.
  - Chia flow mô hình đoán benign thành cụm; sàng mỗi cụm bằng 1 nhãn ngẫu nhiên.
  - Cụm ra attack được rút tiếp đến đủ r = 4 nhãn ngẫu nhiên; nếu cả 4 đều attack và cận dưới Clopper-Pearson ≥ 0,7 thì cảnh báo cả cụm. Cụm lẫn nhãn thì tách đôi.
  - Không huấn luyện bằng nhãn giả.
- **Pilot harness (3 seed, cumulative F1 / FPR):**
  - CSE: d1 0,61 → 0,94; d5 0,55 → 0,95 / 0,5%.
  - ToN: d1 0,84 → 0,91; d5 0,72 → 0,90 / 5%.
  - UNSW giữ nguyên.
  - Seed tệ nhất vẫn hơn fast lane uniform: CSE +0,21, ToN +0,10.
  - Hierarchical sampling (Dasgupta & Hsu 2008) và uniform+kNN thua: FPR 10–15% trên ToN. Một đại diện mỗi cụm, không kiểm định: F1 tương tự nhưng FPR 9–11%.
- **Runner thật khớp harness** ở d5, seed đầu: CSE 0,47 → 0,94, ToN 0,74 → 0,91.
- **BoT không ổn định:** 13 benign mỗi window; runner và harness cho chiều ngược nhau. Chỉ báo cáo như caveat.
- **Prior art gần:** Dasgupta & Hsu 2008, ILAB (Beaugnon RAID 2017), gom cụm alert (Julisch 2003). Lan truyền trong cùng window phải được trình bày là phân loại theo incident.
- PU/novelty không phải nguồn gain. Cảnh báo PU thô vỡ FPR khi benign drift.
- **Việc tiếp:**
  - chạy stage `verify` trên dev bằng runner (288 run, chạy được cục bộ);
  - baseline IWMS và ARF;
  - nhiều held-out natural hơn;
  - chỉ chạy CIC-IDS2017 held-out một lần khi đã chốt.

## Kết luận đến nay (hướng DASP-v2 cũ)

1. **DASP-v2 không hơn baseline đơn giản.**
   - Trên held-out (NF-CSE, NF-ToN, CIC-IDS2017), Experience Replay (ER) hơn DASP-v2 0,009 Macro-F1 (Holm p 7e-7).
   - Nhưng Bayesian ROPE ±0,01 cho thấy tương đương thực tế: P(rope) 0,77; ARF 0,99.
   - Realign, attributed routing và logit adjustment đều ≈ 0 trên held-out. Prior correction bị dev selection tắt.
   - Lợi thế trên dev đến từ việc overfit NF-BoT.
2. **Mọi thành phần đã có prior art**: GDumb/CBRS, LAS (TMLR 2024), Zhang 2013, Kulinski 2020, E2D2 2026. Không thể viết thành bài "method mới".
3. **Khung khuyến nghị: benchmark/hybrid**, với câu hỏi: dưới ngân sách và độ trễ nhãn thực tế, cơ chế drift-adaptive có hơn replay không? Đích nhắm: Computers & Security, EAAI, IEEE TNSM.
4. **AOC/CND/SOUL bản adapt tụt xa ER**: 0,65 / 0,83 / 0,68 so với 0,93. Cần E7 (tái tạo số đã công bố) để chặn phản biện "adapter cài sai".
5. **Drift tổng hợp (feature shift) gần như không làm hại learner có cập nhật**: mất ≤ 0,007. Chỉ natural drift còn dư địa.

## Pilot cục bộ 2026-10-22 (2 dataset × 2 seed)

- **E4 (bỏ shortcut):**
  - Port là shortcut trên NF-ToN (−0,03 đến −0,04).
  - TTL là shortcut trên NF-UNSW (static −0,04).
  - ER > DASP-v2 ở mọi bộ bỏ, nên kết luận vững. **Đáng chạy đủ (876 run).**
- **E6 (chính sách truy vấn):** **vô dụng.**
  - `uncertainty` −0,04 đến −0,18; mixed/stratified ≈ uniform.
  - Ngân sách bão hòa ở 64–128 nhãn/window.
  - **Đừng chạy lưới chính sách trên server.**
- **Độ trễ nhãn là yếu tố lớn nhất.** ER trên NF-CSE chronological: 0,82 (trễ 0), 0,63 (trễ 1), 0,47 (trễ 5); static 0,42.
- **Nguyên nhân:** attack thật đến theo đợt 1–2 window.
  - Window đầu của attack mới luôn bị sót.
  - Trễ 1 thì sót luôn window thứ hai.
- **Novelty không cần nhãn không bắt được onset ở FPR 1%.** Đã thử kNN so với quá khứ, so với benign quá khứ, và gom cụm. NF-ToN AUROC ≤ 0,5.
- **Thông điệp cho bài:** sai số trên natural drift do *độ trễ nhãn so với độ dài đợt attack* quyết định, không phải thuật toán cập nhật, chính sách truy vấn hay số nhãn.

## Việc tiếp theo (người dùng chưa chọn)

- **A.** Rút gọn E6 thành sweep độ trễ và ngân sách (chỉ uniform; ER, DASP-v2, ARF, static; thêm độ trễ 2; đủ 4 stream chronological, khoảng 1.000 run). Pilot cục bộ trước, rồi chạy server cùng E4 đủ.
- **B.** Nghiên cứu tiếp cục bộ: giảm độ trễ tính theo số flow (window nhỏ hơn) hoặc cách bắt onset attack mới.
- **Còn thiếu cho Q1:**
  - E1: metric vận hành, TPR@FPR 0,1%/1% (FPR hiện 4,9–6,4%).
  - E2: natural drift rộng hơn.
  - E3: baseline đơn giản.
  - E5: LODO (đang dở; code có trong worktree).
  - E7: tái tạo số đã công bố của adapter.
  - Bản thảo khung hybrid.

## Lưu ý kỹ thuật

- Chạy nhiều shard song song trên Windows: các shard cùng ghi `<stage>_plan.json` và có thể lỗi `PermissionError`. Chạy lại shard đó là xong; Linux không bị.
- Trên máy này mỗi run mất khoảng 30 s/tiến trình; máy có 12 nhân.
- Commit `a594739c` (LODO, E4/E6/E15) đã push lên `dasp-v2-infra`; phần fast lane/`verify` chưa commit.
