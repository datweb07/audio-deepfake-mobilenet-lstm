# R08 — biên bản đóng băng

**R08_PROTECTED_FILES_UNCHANGED = TRUE.** Đối chiếu SHA-256 trước và sau sửa bản thảo cho 405 tệp, gồm 18 tệp riêng thuộc R08 và 387 tệp dùng chung hoặc bằng chứng khoa học. Không có tệp thiếu, byte thay đổi hay artifact R08 mới. Git HEAD giữ nguyên `917ee0127ba6077903f3258b096f8cf9a2f8f3c7`. Working tree sạch tại thời điểm chụp manifest. Xem [manifest](R08_FREEZE_MANIFEST.json) và [xác minh](R08_FREEZE_VERIFICATION.json).

| Phạm vi | Ghi nhận |
| --- | --- |
| `IMMUTABLE_R08_CODE` | Hai module R08 và test tương ứng. |
| `IMMUTABLE_R08_CONFIG` | Release, seed và chính sách nằm trong mã và hồ sơ hiện có. Thư mục config protocol sinh lúc khóa chưa tồn tại trong snapshot. |
| `IMMUTABLE_R08_APPROVAL` | Hai JSON phê duyệt và mẫu, văn bản xác nhận quyền, snapshot tài nguyên kèm phê duyệt. |
| `IMMUTABLE_R08_COMPLETED_EVIDENCE` | Metadata, qualification, tài nguyên, hồ sơ integrity, tài liệu quyền và hướng dẫn. |
| Shared dependencies | Các source, runner, config, test, manifest và bằng chứng mô hình liệt kê trong manifest. FIX01 cũng được bảo vệ riêng. |
| `R08_RUNTIME_GENERATED_ARTIFACTS` | Các artifact chính thức mà tác giả tạo sau snapshot được phép xuất hiện theo quy trình hiện hữu. Manifest bảo vệ nội dung đã liệt kê, không thiết lập ACL chỉ đọc. |

R08 giữ In-the-Wild revision `eee168f92c367f8c82ff2cf42b6f61e362fd6211`, seed lấy mẫu `20261112`, P0 và ngưỡng 0.5. Qualification lưu lịch sử `DEVELOPMENT_REPLAY_PASS`. Hồ sơ quyền và phê duyệt data access đã có. Hồ sơ R08 chưa có WAV, lock hoặc score tại thời điểm audit. Không chạy runner, replay, tải audio, inference hay training trong lần sửa FIX02.

Tác giả có thể tiếp tục theo [hướng dẫn hiện hữu](../R08_AUTHOR_EXECUTION_GUIDE.md). Dùng phê duyệt data access đã lưu. Kiểm tra tài nguyên và integrity, rồi chạy tuần tự `pilot`, `acquire`, `lock`, `scoring-template`. Tự rà protocol và ký phê duyệt scoring riêng trước `score`, `verify` và `integrity`. Giữ hồ sơ phê duyệt cá nhân và audio ở máy local. FIX02 độc lập với tiến trình R08.

