# Giai đoạn hạ tầng DASP-v2: kết quả

Nhánh `dasp-v2-infra`, tách từ `codex/dasp-original-review-fixes` (d702bc4f). Ngày 18/10/2026. Kế hoạch tổng thể ở [PREREGISTRATION_DASP_V2.md](PREREGISTRATION_DASP_V2.md).

## 1. Chạy song song trên CPU

- `tools/run_original_review.py` có thêm `--shard K/N` và `--threads` (mặc định 1). Shard được gán theo **cả cell ghép cặp** (dataset × scenario × stream seed × model seed). Vì vậy đối chứng event-matched luôn chạy sau run native của cùng cell, trong cùng một tiến trình.
- Mỗi shard ghi `{stage}_completed.shardKofN.json`. File plan dùng chung được ghi atomically. `analyze_original_review.py` gộp các shard, và báo lỗi khi thiếu shard, khi trộn số shard khác nhau, hoặc khi tồn tại cả file unsharded lẫn file shard.
- `tools/launch_shards.sh N ...` chạy N tiến trình CPU đơn luồng; mỗi shard ghi log riêng.
- Nên dùng CPU: mô hình khoảng 20 nghìn tham số, mỗi run 2–6 giây, kết quả CPU tái hiện bitwise. Kết quả GPU đã lệch CPU khoảng 6e-4 Macro-F1, cùng cỡ với hiệu ứng cần đo.

```bash
tools/launch_shards.sh 16 --stage primary --prepared-root /data/prepared \
    --output-root /results/primary --execute --resume
python tools/analyze_original_review.py --stage primary --output-root /results/primary
```

## 2. Baseline

Bản cài đặt cũ giữ nguyên để tái hiện kết quả lưu trữ. Bản sửa mang ID mới.

| ID mới | Lỗi của bản cũ | Cách sửa |
|---|---|---|
| `online_ewc_v2` | (a) Fisher = bình phương gradient trung bình batch: gần cực tiểu, gradient từng mẫu triệt tiêu nhau nên Fisher ≈ 0. (b) Điểm neo đặt lại sau mỗi cửa sổ, mỗi cửa sổ chỉ có khoảng 2 bước AdamW; bước đầu bắt đầu đúng tại điểm neo, nơi penalty có gradient 0. Hệ quả: 68,6% run EWC lệch Online MLP dưới 1e-4. | Fisher thực nghiệm = trung bình bình phương gradient từng mẫu (`torch.func.vmap`). Củng cố điểm neo mỗi `ewc_anchor_interval` cửa sổ (mặc định 4; **phải chỉnh ở pha dev** cùng `ewc_lambda`). |
| `cdda_md_v2` | (a) Đọc `row['accuracy']` tính trên toàn cửa sổ (4.096 nhãn), vượt hợp đồng 128 nhãn. (b) Sau mỗi drift, ngưỡng chỉ tăng lên `error + min_increase`, không bao giờ hạ, nên detector có thể không báo động được nữa. (c) Severity tính sau khi đã đổi ngưỡng nên luôn bằng 0. | Lỗi tính trên 128 nhãn đã truy vấn. Hiệu chuẩn warmup dùng cùng cỡ mẫu 128. Sau drift reset theo kiểu DDM: `cdda_recalibration_windows` cửa sổ kế tiếp ước lượng lại ngưỡng. Severity tính trước khi reset. |
| RepShield | Cảnh báo PCA chỉ ghi log; việc train không phụ thuộc cảnh báo. | **Chưa sửa và loại khỏi so sánh chính.** Chỉ tìm được metadata bài báo (Computer Networks 283, 112308, 2026), không có mô tả phương pháp đầy đủ hay mã tác giả, nên không thể sửa mà không đoán. Nếu có PDF đầy đủ thì sẽ cài đặt lại. |

## 3. Tái hiện stream từ dữ liệu thô

| Dataset | Kết quả |
|---|---|
| NF-UNSW-NB15-v3 (S0, S1-Sudden; 5 seed) | **Không tái hiện được.** `y`, `phases`, `subtypes`, `feature_names` trùng khớp; nhưng `x`, `source_ids`, `timestamps` và các tham số chuẩn hóa khác. Manifest lưu trữ ghi `directional-session-hash`, còn builder trong Git dùng `whole-directional-session deterministic load balancing`. Stream lưu trữ (30/07/2026 09:44) được tạo bởi một phiên bản builder sớm hơn, không còn trong repo. |
| NF-BoT-IoT-v3 (10 scenario × 5 seed, chạy trong một lệnh như bản gốc) | **Không tái hiện được.** Cả 50 manifest trùng hoàn toàn từng trường với bản lưu trữ, `y` cũng trùng, nhưng flow được chọn (`x`, `source_ids`, `timestamps`) khác. Nguyên nhân chưa xác định: có thể builder bị sửa sau khi tạo stream mà manifest không ghi lại, hoặc phiên bản numpy/pandas lúc đó khác (môi trường gốc `.venv` hiện đã hỏng). Lưu ý: chạy chỉ S0 thì manifest còn khác ở `observed_required_family_counts`, vì builder chỉ quét các họ tấn công của scenario được chọn. |
| Tính xác định của builder hiện tại | **Đạt.** Chạy hai lần NF-UNSW (S0, S1; 5 seed) cho 10/10 stream giống hệt nhau. Môi trường: Python 3.12.7, numpy 1.26.4, pandas 2.3.3, scipy 1.16.2, scikit-learn 1.7.2. |
| NF-CSE-CIC-IDS2018-v3 | Manifest lưu trữ tạo lúc 09:49, nhiều khả năng cùng phiên bản builder cũ như UNSW. |
| CIC-IDS2017 | **Builder không còn.** Manifest protocol 2.0 ("exact-feature hash × 10000-row block"); không tìm thấy mã trong repo hay trong `.codex`. |

Hệ quả theo kế hoạch (mục 2): toàn bộ stream của DASP-v2, cả seed dev 2001–2003 lẫn seed xác nhận mới, phải được **tái sinh bằng builder có trong Git**, bằng một lệnh ghi trong tài liệu, rồi dùng chung cho mọi phương pháp. CIC-IDS2017 cần builder mới (phiên bản protocol mới, có test), hoặc phải thay dataset. Kết quả lưu trữ cũ chỉ tái hiện được từ NPZ lưu trữ, không tái hiện được từ dữ liệu thô.

## 4. Kiểm chứng

- `tests/test_dasp_v2_infra.py`: 11 test. Gồm Fisher khớp vòng lặp từng mẫu; Fisher cũ bằng 0 khi gradient triệt tiêu còn bản mới là 0,5; detector mới báo động lại sau khi hồi phục còn detector cũ thì không; CDDA-v2 chỉ thấy đúng 128 nhãn truy vấn; EWC-v2 với Fisher đúng kìm cập nhật; shard phủ đủ job, không trùng cell; chạy 2 shard rồi gộp phân tích.
- Toàn bộ test suite: xem mục 5.

## 5. Kết quả test suite đầy đủ

Windows, `.venv-revision` (Python 3.12.7, PyTorch CPU), có junction chỉ đọc tới dữ liệu ở workspace gốc:

- **174 passed, 1 skipped, 0 failed.**
- Không chạy được 3 test `river_*` (`river_logreg`, `river_hat`, `river_arf`) vì môi trường thiếu gói tùy chọn `river`. Trước khi dùng ARF làm baseline phải cài `river` và kiểm tra lại đường chạy River: audit F06 đã ghi warmup bị cắt còn 128 mẫu và việc chọn mẫu cân bằng nhãn đọc nhiều nhãn hơn mức ghi log.
- Bỏ qua `tests/test_analysis_statistics.py` vì môi trường thiếu `matplotlib`.

## 6. Việc còn lại trước pha dev

1. Quyết định CIC-IDS2017: viết builder mới hoặc thay dataset.
2. Tái sinh toàn bộ stream (seed dev 2001–2003 và seed xác nhận 3001–3005) bằng builder trong Git, trong một lệnh có ghi tài liệu. Kiểm tra seed xác nhận rời rạc session với seed dev.
3. Cài và audit đường chạy River/ARF dưới hợp đồng 128 nhãn.
4. Chuyển so sánh chính sang hợp đồng 128 nhãn: SSF dùng `ssf_budget`, DASP dùng `label_mode='matched'`, thêm `online_ewc_v2`/`cdda_md_v2` vào danh sách phương pháp của stage mới.
