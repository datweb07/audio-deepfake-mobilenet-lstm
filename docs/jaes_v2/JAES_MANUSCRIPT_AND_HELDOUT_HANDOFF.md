# Bàn giao bản thảo JAES và chuẩn bị held-out

Ngày 10/11/2026. Trạng thái khoa học `JOURNAL_CONTRIBUTION_INSUFFICIENT` để nộp ngay. Hai cặp phát triển đủ cho bước held-out đã đăng ký. `SEED_9307_NO_GO`, medium-plus và foundation NO-GO. C2 vẫn bị loại ở mức chẩn đoán và C3 deferred.

## Bản thảo và kiểm toán

Bản LaTeX có abstract, introduction, related work, formulation, methods, setup, results, discussion, limitations và conclusion. Bốn bảng kết quả và bốn hình vector được dựng từ JSON hai seed, cùng bảng cấu hình và so sánh prior art. Hình gồm so sánh aggregate, thay đổi lỗi A01–A06, false alarm theo speaker và phân bố chênh score giữa hai thứ tự. Phân tích giữ A05/A06 bất lợi, chuyển chiều clean B và giới hạn của hai seed trên cùng cohort. Nguồn định lượng và phạm vi mỗi claim có trong claim map. Không có kết quả held-out trong bản thảo.

Tài liệu tham khảo phân biệt preprint, proceedings, bài tạp chí và chuẩn Opus. Audit ghi rõ bản ghi gốc và mức kiểm tra nội dung. Các mục xác minh bằng metadata/abstract chỉ hỗ trợ mô tả phạm vi chung. Exact venue, bibliographic citation và quyền tái sử dụng conference còn cần tác giả xác nhận. Tệp DOCX khớp SHA-256 `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073`. [Hồ sơ LaTeX](../../papers/jaes_v2/jaes_manuscript/) · [Mở rộng conference](../../papers/jaes_v2/jaes_manuscript/CONFERENCE_EXTENSION_DISCLOSURE.md)

Bốn run được audit qua contract, seal, checkpoint được chọn và SHA-256 model xuất. Các checkpoint lần lượt là 560/760 cho 9107 và 520/720 cho 9207. Fingerprint trainer giữ nguyên. Prelock cũ cũng giữ nguyên. Runner scoring mới có SHA-256 mới do bổ sung cổng phê duyệt và đọc danh sách mẫu đã khóa. Công thức signal transform, score và bootstrap giữ nguyên. Khi tạo proposed lock, tool ghi rõ mã nguồn auditor và phiên bản runner mới.

Các báo cáo cũ từng đề xuất ba cặp medium. Protocol đã khóa yêu cầu 9107/9207, và tổng hợp hậu 9207 đã chốt 9307 NO-GO. Hồ sơ này dùng quyết định mới nhất và giữ nguyên các báo cáo lịch sử.

QA tĩnh kiểm tra 25 citation keys và các đường dẫn input, nhãn tham chiếu, abstract 138 từ và bốn PDF hình. 22 test pytest đạt, gồm ZIP giả lập cho cổng phê duyệt, nguồn trùng, thiếu trial/license, metadata lỗi, sai hash, sampling/fallback, scoring bị từ chối, cùng test waveform development và technical resume. Biên dịch toàn bài và kiểm tra dàn trang còn chờ bộ class chính thức và LaTeX.

## Quy trình hai phê duyệt

**Chưa có phê duyệt.** Chưa đọc official CM protocol, license, audio payload hay outcomes. Dùng bản prelock để đối chiếu SHA-256. Tài liệu này chỉ mô tả schema. Tác giả tự cung cấp bản ghi phê duyệt, với `author`, `approved_at` và `approval_reference` thực tế.

**Bước 1.** Sau khi tác giả cấp `AUTHOR_APPROVAL_PROTOCOL_READ`, file `protocol_read_author_approval.json` phải có `scope` tương ứng, `prelock_sha256`, `allow_selected_audio_byte_hashes: true`, `subset` là `primary` hoặc `fallback`, và `resource_subset_reason`. Hash byte chỉ áp dụng cho nguồn được chọn, không decode. Sau đó chạy từ root repo:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.heldout_protocol_audit --protocol-approval outputs/jaes_v2/phase10/empirical_execution_v1/protocol_read_author_approval.json --output outputs/jaes_v2/phase10/empirical_execution_v1/heldout_proposed_9107_9207_v1.json
```

Tool kiểm tra unique trial và speaker/attack metadata, đối chiếu mọi trial ID với ZIP member, đọc license, ghi số member nằm ngoài protocol, lấy mẫu speaker round robin theo seed 91109, giữ shortage và kiểm tra trùng ID, speaker, SHA-256 theo các manifest identity đã khóa. Protocol năm cột không có trường lineage đủ để xác minh generator. Phải xem license, shortage và mức độ hỗ trợ trước khi phê duyệt scoring. Nếu khác số trial công bố hoặc không tìm thấy member, tool dừng. Chỉ SHA-256 của nội dung FLAC được tính trước scoring.

**Bước 2.** Sau khi xem proposed lock, tác giả có thể cấp `AUTHOR_APPROVAL_HELDOUT_SCORING` trong file `heldout_scoring_author_approval.json` với `scope`, `proposed_lock_sha256`, `selected_rows_sha256` và `license_sha256`. Chỉ chạy lệnh sau khi có phê duyệt này và resource gate PASS:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.empirical_heldout --proposed-lock outputs/jaes_v2/phase10/empirical_execution_v1/heldout_proposed_9107_9207_v1.json --scoring-approval outputs/jaes_v2/phase10/empirical_execution_v1/heldout_scoring_author_approval.json
```

Bước này dùng danh sách nguồn đã duyệt, xác thực artifact và byte hash khi chấm, và giữ các shard khi dừng kỹ thuật. Cùng hai file cho cùng execution identity khi resume. Run hoàn tất từ chối ghi đè. Lệnh cũ `--approved-lock` trong báo cáo trước đã được thay bằng hai đối số trên.

## Tài nguyên và mốc

Primary 1.088 nguồn / 17.408 prediction là lựa chọn dự kiến theo benchmark development khoảng 50 phút, chưa tính I/O, audit hoặc bootstrap. Fallback 544 nguồn / 8.704 prediction chỉ hợp lệ nếu ngân sách đã đo không đủ cho primary trước khi mở nhãn. Cả hai dùng bốn model và cùng ngưỡng RAM, commit và disk. Giảm số nguồn không xử lý thiếu commit lúc khởi tạo.

Snapshot 10/11 ghi RAM khả dụng 2,395 GiB, commit khả dụng 2,168 GiB, C còn 7,918 GiB, D còn 19,350 GiB, RSS tiến trình 0,095 GiB. Thiếu 0,832 GiB commit để đạt ngưỡng 3 GiB. `SYSTEM_COMMIT_PRESSURE` chặn scoring với cả hai ngân sách tại thời điểm đo. Kiểm tra lại ngay trước khi chạy. [Resource snapshot](../../outputs/jaes_v2/phase10/empirical_execution_v1/jaes_heldout_resource_check_v1.json)

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_experiment resources
```

10–14/11 hoàn tất source và fixture, bổ sung build chính thức. 15–18/11 giải quyết resource và phê duyệt protocol. 19–25/11 chấm held-out nếu được duyệt. 26–30/11 phân tích và chèn kết quả đã xác minh. 01–10/12 hoàn chỉnh bài. 11–14/12 QA. 15–20/12 hoàn tất hồ sơ, quyền tái sử dụng và disclosure. Bằng chứng còn thiếu quan trọng nhất là đánh giá held-out khóa sẵn, cùng với diễn giải giới hạn same-release và đối chiếu prior art.
