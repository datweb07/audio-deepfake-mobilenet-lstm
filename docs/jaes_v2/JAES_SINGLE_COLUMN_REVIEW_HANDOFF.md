# Bàn giao bản review một cột và held-out

10/11/2026. Bản gửi giảng viên gồm 24 trang A4 một cột, chữ 11 pt, lề 25 mm và giãn dòng thân bài 1,5. Trang 1–21 là phần trước tài liệu tham khảo, trang 22 có phần disclosure và bắt đầu tài liệu tham khảo, trang 23–24 tiếp tục danh mục. Thân bài khoảng 7.583 từ theo phép đếm source bỏ float, công thức và lệnh trích dẫn. Có 26 nguồn được trích dẫn, bốn hình vector và bảy bảng. Không có lỗi fatal, citation hoặc reference chưa phân giải, ký tự thiếu hay overfull box trong log review cuối. Đã render và rà bố cục đủ 24 trang. Bản ngắn hơn một trang so với mốc 25, và không chèn nội dung đệm.

[PDF giảng viên](../../papers/jaes_v2/jaes_manuscript/build/LAVA_JAES_Review_Draft.pdf) · [Source và hướng dẫn build](../../papers/jaes_v2/jaes_manuscript/README_BUILD.md) · [Ma trận phản biện nội bộ](../../papers/jaes_v2/jaes_manuscript/CONFERENCE_TO_JOURNAL_REVIEW.md)

Hai entry point dùng chung nội dung. Class JAES nguyên bản giữ bố cục tạp chí. Bản tương thích của style BibTeX xử lý hai lỗi parse của style được cung cấp và giữ bản style gốc để đối chiếu. Xem README để chọn đúng entry point trong Overleaf. Phí Research Paper theo AES là 0 USD cho 10 trang journal đầu và 100 USD mỗi trang từ trang 11. Dùng số trang journal sau dàn trang chính thức để dự toán. Số trang của bản review một cột không dùng để tính phí. [AES page charges](https://aes.org/publications/open-access-publications/)

Bản JAES hai cột build thành công **16 trang**, với 14 trang trước tài liệu tham khảo (văn bản đến trang 12, hai trang tiếp theo có float) và hai trang tài liệu tham khảo. Đã kiểm tra bố cục render và log. Bản class giữ nguyên, wrapper bỏ khung tiểu sử rỗng đến khi có metadata thật và đưa các float ra trước tài liệu tham khảo. Hai build đều có cảnh báo thiếu tháng ở 25 mục BibTeX. Dự toán nếu bản xuất bản giữ đúng 16 trang là **600 USD phí vượt trang**, chưa gồm APC tùy chọn hoặc thay đổi sau dàn trang. [PDF bố cục JAES](../../papers/jaes_v2/jaes_manuscript/build/LAVA_JAES_Official_Layout.pdf) · [Chính sách phí AES](https://aes.org/publications/open-access-publications/)

## Rà held-out

Proposed lock khớp SHA-256 `f422a0a20580bdca4c4a99c6aea58add2958edf3e7efabb50f57c274496f23a8`. Prelock, approval đọc protocol, selected-row hash, license hash, bốn model, bốn điều kiện, quota và non-archive bindings khớp. Kiểm tra lại overlap ID, speaker và SHA-256 byte trên ba manifest cho kết quả 0. Giữ 1.088 nguồn và 17.408 dự đoán dự kiến. Không có output scoring. Audit kỳ này chỉ đọc lock và manifest, không mở ZIP official. `authorized: false` và `HELDOUT_NOT_AUTHORIZED`.

[Mẫu scoring chưa phê duyệt](../../papers/jaes_v2/jaes_manuscript/HELDOUT_SCORING_APPROVAL_TEMPLATE.json) gắn đúng ba hash. Sau khi rà lock và license, tác giả điền tên, thời điểm và tham chiếu phê duyệt thật, rồi lưu bản hoàn chỉnh vào folder evidence dưới tên `heldout_scoring_author_approval.json`. Không điền nếu chưa chính thức phê duyệt. Chạy preflight trước. Snapshot lúc 19:26:22 +07:00 báo `RESOURCE_BLOCKED`, với 0,59 GiB RAM và 1,02 GiB commit khả dụng, so với ngưỡng 2 và 3 GiB. Ổ C còn 4,42 GiB và D 19,58 GiB. Giải phóng RAM/commit bằng cách đóng ứng dụng đang sử dụng bộ nhớ, sau đó đo lại. Không đổi subset để vượt ngưỡng.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_experiment resources
```

**NOT_AUTHORIZED_FOR_EXECUTION** cho đến khi có phê duyệt scoring riêng và preflight mới PASS.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.empirical_heldout --proposed-lock outputs/jaes_v2/phase10/empirical_execution_v1/heldout_proposed_9107_9207_v1.json --scoring-approval outputs/jaes_v2/phase10/empirical_execution_v1/heldout_scoring_author_approval.json
```

## Mức sẵn sàng

Bản review đủ để góp ý khoa học. Nghiên cứu mở rộng conference có đối chứng mới ở hai block, kết quả lớp/attack/speaker và chẩn đoán thứ tự. Hai block không thiết lập độ ổn định trên quần thể seed. Dữ liệu hiện tại không chứng minh generalization ngoài corpus, cơ chế forensic độc lập, hiệu quả ngoài hiện trường, hoặc thuật toán mới. Chưa sẵn sàng nộp JAES. Trước 20/12, ưu tiên duyệt và thực thi held-out khóa sẵn khi đủ tài nguyên, đưa kết quả đã kiểm chứng vào bài, rồi chốt author metadata, DOI/venue và quyền conference. Không cần train thêm.
