# Tiền đăng ký 1 (commit trước mọi lượt TN2)

Ngày: 2026-09-25. Tài liệu này chốt các ngưỡng, quy tắc quyết định và mã phân tích **trước khi** có bất kỳ kết quả nào của các phép thử nguyên nhân. Mọi thay đổi sau mốc này được ghi vào `SO_GHI_CHEP.md` kèm lý do. Kết quả sẽ được báo cáo theo cả phiên bản cũ lẫn phiên bản mới.

## 1. Ngưỡng và quy tắc (Bảng 8, Bảng 9 của đề cương sửa, Phần 4 và 6)

Mã: `phan_tich/thong_ke.py`, kiểm thử: `phan_tich/kiem_thu.py` (11/11 đạt), bộ tính AP: `phan_tich/ap_nhanh.py` (khớp COCOeval tới 4 chữ số thập phân trên UODD và ArTaxOr).

| Hằng số | Giá trị |
|---|---|
| Sụp đổ | R ≤ 0,2 |
| Hồi phục | Cận dưới KTC của R ≥ 0,9 |
| Tốc độ học không tầm thường | η_b ≥ 10^-2 · η_h |
| δ_R (quy tắc ba trạng thái) | 0,25; độ nhạy ở 0,15 và 0,35 |
| KTC | 95%, phân vị, bootstrap ghép cặp hai tầng (lần lấy mẫu, ảnh), B = 2.000 |
| δ_rel | max(2·CV̄; 0,05), CV̄ = CV trung bình của A (SEED 0) qua s0–s4 trên ba bộ chọn tham số. Giá trị ghi bằng commit bổ sung sau TN1 |
| Tính lặp lại đủ tốt | SD do khởi tạo (SEED 0, 1, 2 trên s0) ≤ 0,5 × SD giữa các lần lấy mẫu |

Quy tắc ba trạng thái, tổng hợp nhóm K, bảng quyết định Q0–Q6, trạng thái cải thiện và kết luận tổng hợp được cài đúng như `thong_ke.py`.

## 2. Điều chỉnh thiết kế do đọc mã và dữ liệu (so với đề cương sửa)

1. **η_h và số bước theo từng bộ,** theo tệp cấu hình công bố:
   - η_h = 0,002, trừ DeepFish (0,001);
   - số bước ở 10-shot: 200 cho ArTaxOr và NEU-DET, 500 cho Clipart1k và DIOR, 200 cho DeepFish, 100 cho UODD.

   Lưới η_b luôn tính theo tỉ lệ r = η_b/η_h ∈ {10^-3; 10^-2,5; 10^-2; 10^-1,5; 10^-1; 10^-0,5; 1}.
2. **Mạng đề xuất vùng là ResNet-50 riêng**, không nhận đặc trưng DINOv2. Nhánh K1 chỉ còn kiểm "cập nhật mạng đề xuất vùng với tốc độ học chung" (C0 so với C). C0 cần mã tính mất mát mạng đề xuất vùng, đang viết. TN12-K1 chạy sau khi mã này được kiểm.
3. **Vectơ đại diện lớp và nền là tham số học được**, khởi tạo từ phần trích đặc trưng ban đầu, không bao giờ được tính lại. K2 giữ nguyên thiết kế.
4. **Mã gốc chạy DINOv2 dưới autocast fp16, không có GradScaler.**
   - TN2 dùng bf16 làm mặc định, theo đề cương sửa.
   - TN12-K3 so fp16 (như gốc) với bf16, kèm nhật ký P4: chuẩn gradient theo nhóm và số tham số có gradient không hữu hạn ở mỗi bước.
5. **DINOv2 luôn ở chế độ eval** (mã gốc gọi `backbone.eval()` mỗi bước; `drop_path_rate` = 0). Đề tài giữ nguyên hành vi này khi mở.
6. **SEED cố định bằng 0** cho mọi lượt dùng trong phép so ghép cặp. Các lượt TN1 s0 với SEED mặc định (−1) chỉ dùng để kiểm tái lập so với số đã công bố.
7. **Lần lấy mẫu tự sinh** theo quy tắc "mỗi ảnh một hộp" của tệp chính thức (`ma_nguon/sinh_lan_lay_mau.py`).

## 3. Cấu hình của TN2 (lưới thô)

- **Cấu hình D(r):** mở toàn bộ phần trích đặc trưng (24 khối, lớp nhúng, chuẩn hoá cuối), mạng đề xuất vùng giữ nguyên, vectơ đại diện theo cơ chế gốc.
- **Chế độ và tối ưu:** bf16, SGD theo lịch công bố của từng bộ, lô hiệu dụng 16, gradient checkpointing.
- **Phạm vi:** 7 mức r × 3 bộ chọn tham số (ArTaxOr, Clipart1k, NEU-DET) × lần lấy mẫu s0–s2 × SEED 0.
- **P3:** lưu trọng số tại 5 mốc trên s0; mốc tỉ lệ theo lịch của từng bộ (1/8, 1/4, 1/2, 3/4, 1).

## 4. Mã phân tích tại mốc này

Mã băm commit được ghi trong `git log` tại commit có thông điệp "Tien dang ky 1".
