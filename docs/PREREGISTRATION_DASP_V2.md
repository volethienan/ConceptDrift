# Kế hoạch thực nghiệm khai báo trước — DASP-v2

Bản nháp ngày 18/10/2026. Trạng thái: **chưa khóa**. Tài liệu chỉ có hiệu lực khai báo trước khi được commit kèm hash và chưa có kết quả nào của giai đoạn xác nhận được xem.

## 1. Định vị bài và các khẳng định

Khẳng định chính (B): DASP là cơ chế thích nghi cục bộ ở mức tham số, kiểm soát và kiểm toán được, đạt chất lượng không kém cập nhật toàn bộ dưới cùng ngân sách nhãn, đồng thời giữ kiến thức cũ tốt hơn hoặc có chi phí cập nhật thấp hơn.

Khẳng định phụ (A) chỉ được viết khi qua tiêu chí ở mục 6:
- **A-framework:** DASP-v2 dự đoán tốt hơn các baseline đã sửa, dưới cùng hợp đồng 128 nhãn.
- **A-routing:** routing của DASP-v2 tốt hơn routing đơn giản và tốt hơn cập nhật toàn bộ module.

Thuật ngữ: "module" là nhóm tham số vận hành, không phải nhóm đặc trưng có ý nghĩa ngữ nghĩa hay nhân quả. Không dùng "attribution" theo nghĩa giải thích nhân quả.

## 2. Dữ liệu và phân vai

| Vai trò | Dữ liệu | Được dùng để |
|---|---|---|
| **Phát triển (dev)** | NF-BoT-IoT-v3, NF-UNSW-NB15-v3; 10 scenario; stream seed 2001–2003; model seed 91 | Thiết kế và chọn cấu hình DASP-v2, chỉnh tham số baseline |
| **Xác nhận nội bộ, dataset giữ lại** | CIC-IDS2017, NF-CSE-CIC-IDS2018-v3, NF-ToN-IoT-v3 | Kiểm định H1–H5 sau khi khóa |
| **Xác nhận nội bộ, seed mới** | Cả 5 dataset, stream seed mới 3001–3005, model seed 91–92 | Kiểm định; báo cáo dataset dev tách riêng khỏi dataset giữ lại |
| **Xác nhận ngoài, chưa từng đụng** | CICIoT2023 (Neto et al., Sensors 2023) | Mở một lần duy nhất sau khi khóa |
| **Chronological tự nhiên** | NF-ToN-IoT-v3 (đã dùng ở vòng trước), NF-UNSW-NB15-v3, NF-CSE-CIC-IDS2018-v3, xếp theo `FLOW_START_MILLISECONDS` | Kiểm tra hành vi khi drift tự nhiên; không gọi là external validation |

Lý do chọn:
- NF-BoT-IoT-v3 có benign rất hiếm (13/4.096 mỗi cửa sổ ở S0). NF-UNSW-NB15-v3 có gate 32 mẫu thường chỉ có 1–4 benign. Hai dataset này bộc lộ đúng các lỗi cần sửa (gate, mất cân bằng lớp), nên dùng chúng để phát triển.
- CICIoT2023 là dataset gần đây, khác phòng thí nghiệm, khác bộ trích xuất đặc trưng, và có các họ tấn công tương thích với scenario hiện tại (DDoS, DoS, Recon…). Trước khi tải cần kiểm tra danh sách cột chính thức. Nếu không có timestamp thì chỉ dùng cho scenario có kiểm soát.
- NF-BoT-IoT-v3 không dùng cho chronological vì benign quá hiếm.

Ràng buộc dữ liệu:
- Stream seed mới phải rời rạc theo session với các stream dev, kiểm tra bằng `source_ids`. Nếu builder không bảo đảm được điều này, kết quả seed mới trên dataset dev chỉ được báo là "độ bền theo seed".
- Lệnh raw → prepared phải chạy lại được. Kiểm tra bằng cách tái sinh `stream_seed2001` của một dataset và so hash với NPZ lưu trữ. Nếu không khớp, ghi rõ và dùng NPZ mới cho mọi phương pháp.
- CIC-IDS2017 có lỗi nhãn đã được công bố (Engelen et al., 2021). Ghi vào phần giới hạn.

## 3. Hợp đồng nhãn (áp dụng cho mọi phương pháp ở phần so sánh chính)

- Mỗi cửa sổ sau warmup: đúng 128 chỉ số được truy vấn đồng nhất, không nhìn nhãn, cùng seed cho mọi phương pháp. Nhãn đến trễ một cửa sổ. Dự đoán luôn xảy ra trước khi nhận nhãn.
- Mọi quyết định (chọn mẫu, gate, alarm, routing) chỉ được dùng 128 nhãn này cộng với nhãn đã đến trước đó trong bộ nhớ. Không phương pháp nào được đọc metric tính trên toàn cửa sổ.
- Warmup giống nhau cho mọi phương pháp.
- DASP giao thức gốc (4.096 nhãn mỗi cửa sổ) chỉ là phân tích độ nhạy, không phải headline.

## 4. DASP-v2: các thay đổi được phép

Tối đa bốn thành phần. Mỗi thành phần có công tắc riêng để ablation.

| ID | Thành phần | Mô tả |
|---|---|---|
| C1 | Gate dựa trên bằng chứng | Mỗi lớp cần tối thiểu `n_min` mẫu. Dùng cận dưới Wilson của recall từng lớp thay cho Macro-F1 cố định. Thiếu mẫu thì gộp nhãn đã đến trong bộ đệm gate có giới hạn; vẫn thiếu thì trạng thái là "unknown" và chỉ chạy background update. |
| C2 | Routing theo lợi ích đo được | Thử một bước cập nhật cho từng module trên tập repair, đo mức giảm loss trên tập gate, chọn top-k theo lợi ích. Không module nào có lợi ích dương thì chỉ cập nhật classifier. Chi phí thử nghiệm được tính vào bộ đếm gradient và vào thời gian chạy. |
| C3 | Xử lý mất cân bằng | Logit adjustment (Menon et al., ICLR 2021), prior lớp ước lượng từ nhãn đã đến. |
| C4 | Classifier nhỏ hơn | Thu nhỏ head dùng chung để tính cục bộ có ý nghĩa về số tham số. Đặt mục tiêu head chiếm ≤ 40% tham số. |

Quy tắc phát triển:
- Chỉ chạy trên phần dev. Ngân sách tìm kiếm tối đa 24 cấu hình cho DASP-v2.
- Quy tắc chọn được khai báo trước: mean-window Macro-F1 trung bình trên các cell dev. Hòa trong phạm vi 0,002 thì chọn cấu hình rẻ hơn.
- Sau khi chọn: commit cấu hình và code, ghi hash vào mục 9. Sau thời điểm này không chỉnh gì thêm. Mọi thay đổi bắt buộc (ví dụ sửa bug) phải ghi vào nhật ký sai lệch và chạy lại toàn bộ.

## 5. Phương pháp so sánh

Baseline, đều dưới hợp đồng ở mục 3:
- Online MLP, Experience Replay, A-GEM.
- Online EWC **đã sửa**: Fisher thực nghiệm theo từng mẫu, cập nhật trực tuyến, chỉnh λ.
- SSF dưới ngân sách 128 nhãn (`ssf_budget`), gọi là "adapted SSF".
- CDDA-MD **đã sửa** (bỏ việc đọc accuracy toàn cửa sổ, sửa ngưỡng alarm bão hòa). Nếu không sửa được trung thực thì bỏ hẳn.
- RepShield: chỉ train khi alarm bật, đúng theo mô tả gốc. Nếu không đủ nguồn để tái hiện thì bỏ hẳn.
- ARF (River): baseline streaming chuẩn, cùng 128 nhãn.

Chỉnh tham số công bằng: mỗi baseline được tối đa 24 cấu hình trên cùng phần dev, cùng quy tắc chọn như DASP-v2. Không tuyên bố tái hiện SOTA gốc. Mọi bản cài đặt đều là "adapted".

Đối chứng nội bộ, dùng cùng lịch sự kiện (event-matched) với DASP-v2:
- all-module, classifier-only, random top-k, fixed top-k.
- routing của DASP gốc (MMD + conflict).
- k = 1, 2, 3.
- flat-all: mạng tuần tự cùng tổng số tham số.
- Ablation từng thành phần C1–C4.

## 6. Giả thuyết và tiêu chí

Đơn vị phân tích: một cell dataset × scenario, lấy trung bình qua các seed. Các seed không được coi là quan sát độc lập.

| ID | Giả thuyết | Tiêu chí đạt (phải thỏa tất cả) |
|---|---|---|
| H1 (A-framework) | DASP-v2 hơn baseline tốt nhất | ΔMacro-F1 ≥ +0,005; cận dưới CI 95% của bootstrap phân cấp > 0; p Wilcoxon hai phía sau Holm < 0,05; benign FPR tích lũy không tệ hơn quá 1 điểm %; chênh dương ở ít nhất 4/5 dataset |
| H2 (A-routing) | Routing v2 hơn all-module và random | Với từng đối chứng: Δ ≥ +0,003, cận dưới CI > 0, Holm p < 0,05 |
| H3 (B, không kém hơn) | v2 không kém all-module | Cận dưới CI 95% của Δ > −0,005 |
| H4 (B, giữ kiến thức) | Cục bộ giữ kiến thức cũ tốt hơn all-module | Trên tập test cố định của các regime cũ (S4-Recurrent, S6-Combined): mức giảm F1 thấp hơn, cận dưới CI > 0 |
| H5 (B, chi phí) | Cục bộ rẻ hơn all-module | Số tham số thay đổi, số phần tử gradient được yêu cầu (kể cả phần thử nghiệm của C2) và thời gian cập nhật end-to-end đều thấp hơn |

Quy tắc viết bài:
- H1 hoặc H2 không đạt: không dùng từ "outperform/superior" cho giả thuyết đó, và báo cáo kết quả âm.
- H3 không đạt: khẳng định B cũng phải hạ xuống thành báo cáo mô tả.

## 7. Thống kê

- Bootstrap phân cấp (lấy mẫu lại dataset, rồi scenario trong dataset), 10.000 lần, seed cố định.
- Wilcoxon hai phía trên các cell. Holm riêng trong từng họ (H1: các baseline; H2: các đối chứng routing).
- Phân tích độ nhạy theo dataset. Với chỉ 5 dataset, kiểm định theo dataset có p nhỏ nhất là 0,0625, nên chỉ dùng để mô tả.
- Báo cáo đầy đủ: mean-window và cumulative Macro-F1, attack recall, FNR, benign FPR (tách theo-run và gộp), thời gian, bộ nhớ.
- Thiếu cặp thì dừng và báo cáo, không âm thầm bỏ cặp.

## 8. Thứ tự thực hiện

1. **Hạ tầng:** sharding cho runner và chạy trên CPU (mô hình khoảng 20 nghìn tham số; GPU cho kết quả khác CPU 6e-4, cùng cỡ với hiệu ứng cần đo). Sửa baseline. Kiểm tra tái hiện builder.
2. **Phát triển** trên phần dev; chọn và khóa DASP-v2 cùng các baseline.
3. **Khóa:** commit, ghi hash vào mục 9; nên đăng ký kèm timestamp trên OSF.
4. **Xác nhận nội bộ** (cell dataset giữ lại và seed mới).
5. **Xác nhận ngoài** trên CICIoT2023 và các stream chronological.
6. **Phân tích và viết bài** theo quy tắc ở mục 6.

Ước lượng khối lượng giai đoạn 4: 500 cell × khoảng 17 điều kiện ≈ 8.500 run. Mỗi run khoảng 2–6 giây trên CPU, tức khoảng 10 giờ đơn luồng, chia được cho nhiều nhân.

## 9. Khóa

| Mục | Giá trị |
|---|---|
| Commit code + cấu hình DASP-v2 | _(điền khi khóa)_ |
| Hash cấu hình baseline | _(điền khi khóa)_ |
| Hash NPZ xác nhận | _(điền khi khóa)_ |
| Ngày khóa | _(điền khi khóa)_ |

## 10. Nhật ký sai lệch

_(Ghi mọi thay đổi sau khi khóa: ngày, lý do, phạm vi chạy lại.)_
