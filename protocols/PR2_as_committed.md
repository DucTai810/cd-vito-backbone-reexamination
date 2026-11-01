# Tiền đăng ký 2 — TN11, 28/10/2026

Phạm vi: A-default và công thức dev đã chốt trên DIOR, UODD, DeepFish; s0–s4, seed0. Tổng 30 lượt.
Không dùng AP hoặc dự đoán của ba bộ giữ lại để chọn công thức hay điều chỉnh tham số.
Chi tiết cấu hình, lịch đầy đủ theo từng bộ, trọng số nguồn và hash mã ở protocol.json.
Giữ nguyên mẫu s0 công bố; sinh s1–s4 bằng quy tắc cố định từ train. Tính lại prototype cho cả s0–s4 với seed0, dùng cùng tệp cho hai cấu hình ghép cặp.

Sửa đổi số học: cả hai nhánh dùng T4, FP16 và GradScaler, batch hiệu dụng16 (4×4).
A giữ backbone đông cứng, SGD và LR/weight decay mặc định của cấu hình công bố.
Công thức cuối giữ nguyên AdamW, LR head0.0001, WD0.05, tỷ lệ backbone/head0.01, LLRD0.9 từ dev.
Không tái sử dụng kết quả PC khác môi trường làm đối chứng chính.

Phép so chính của bản giảm: công thức cuối so với A-default, ghép cùng bộ, split và môi trường.
Báo cáo AP từng split, trung bình, SD mẫu, chênh lệch AP và tương đối; bootstrap ghép cặp hai tầng split/ảnh B=2000, seed0.
Ngưỡng tác động tương đối giữ 0.0773 từ TN1; không tính lại bằng dữ liệu giữ lại.
Ba bộ cố định, không suy rộng thành toàn bộ miền; báo độ nhạy bootstrap chỉ theo ảnh vì chỉ có năm split.
Đánh giá theo mã thống kê gốc được lưu cùng bản này. Không tuyên bố vượt A-tuned hoặc SOTA.

Cặp đầu phải hoàn tất cả A và công thức, đủ bước, gradient hữu hạn, có prediction phục vụ thống kê rồi mới mở hàng đợi còn lại.
Lỗi kỹ thuật được sửa và ghi lịch sử, không sửa tham số theo AP. Thiếu quota thì chờ, không tự cắt lịch hoặc split.
Trọng số cuối, cấu hình, log và prediction được giữ trong Kaggle Output; thu result marker có kiểm hash trước khi tính hoàn tất.
