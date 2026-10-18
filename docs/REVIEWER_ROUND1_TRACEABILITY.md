# Reviewer vòng1 → code và thực nghiệm

| Ý kiến | Thực hiện trong nhánh | Cần thực nghiệm/biên tập tiếp |
|---|---|---|
| Routing random/fixed/MMD/conflict/all, cùng repair/replay budgets | Stage routing; reference event schedule; multiset plasticity; log actual parameters; unit tests và CUDA smoke | Chưa chạy full stage trên50dataset-scenario cells và toàn seeds |
| Equal total supervision hoặc giới hạn claim protocol-specific | Main giữ original protocol, ghi actual label exposure và scope; stage fairness DASP/SSF cùng128queried labels; test đổi nhãn ngoài query không làm thay model | Chưa chạy full fairness matrix; không gọi main là equal-label benchmark |
| Tách contribution router/backbone/replay/weighting/gate/class balancing | Stages routing/components; flat-all matched parameter budget vs modular-all; GPU smoke | Chưa chứng minh combined router có predictive advantage |
| k1/k2/k3, head-only và head share | Routing controls và identity-based locality; log optimized/changed/representation/head fractions | Không suy từ parameter invariance thành functional/causal attribution |
| Feature-order robustness | Stage permutation: original +3permutations, cùng streams và seeds | Chưa full run |
| Natural chronological/operational stream | Stage natural nhận NPZ có source IDs/timestamps; preflight global ordering; giữ preparation script | Chưa có untouched new capture được xác minh trong lượt này; không gọi capture cũ là external replication |
| Pilot vs confirmation, reproducibility,4000records | Main plan đúng4000; không reuse pilot archive; config+code+data+environment identity; output receipts; phục hồi tools từ Git | Unseen-setting validation/tuning record vẫn cần; chưa upload archive hay DOI |
| Cluster-aware uncertainty | Strict pairs; dataset-scenario bootstrap + dataset-only sensitivity; two-sided Wilcoxon/Holm; tests thiếu/duplicate pairs | Ít dataset vẫn hạn chế generalization; seed không là deployment độc lập |
| Systems trade-off: adaptation và inference | Common full-cycle timer CUDA barriers; requested-gradient phases; không gọi tensor elements là FLOPs | GPU smoke không phải benchmark tốc độ final; baseline counters có scope cũ, không chia ratio lẫn scope |
| Worst-case/negative findings và operational errors | Operational per-run counts, recall/FNR/FPR; cumulative vs mean-window; gate sensitivity riêng | Chưa full failure analysis; phải giữ negative results khi có |
| Metadata tác giả, funding, COI; rút gọn manuscript | Ngoài scope code/thực nghiệm theo chỉ đạo hiện tại | Không điền giả thông tin tác giả; manuscript chưa sửa trong nhánh này |

Hai lỗi vòng2 liên quan trực tiếp DASP đã được xử lý: gate một lớp và requested gradients/optimizer accounting. Những vấn đề baseline/prototype khác từ audit trước không được âm thầm đưa vào sửa thuật toán DASP gốc.
