# Bàn giao sprint tiền nộp JAES

10/11/2026. Hai seed 9107/9207 với bốn model đã seal là bộ đối chứng hiện hành. 9307, medium-plus và foundation vẫn NO-GO. C1 là phân tích và bằng chứng tái lập, C2 bị loại ở mức chẩn đoán, C3 hoãn. Bản thảo vẫn `JOURNAL_CONTRIBUTION_INSUFFICIENT` để nộp ngay. Đánh giá held-out trong cùng release là khoảng trống thực nghiệm cần xử lý trước 20/12.

## Tài nguyên và lựa chọn đã đăng ký

PASS lúc 12:16:07 +07:00 do tác giả cung cấp có RAM 3.18 GiB, commit 11.11 GiB, C 12.16 GiB và D 19.35 GiB. Lần đo lại lúc 12:39:00 +07:00 có RAM 1.64 GiB, commit 2.37 GiB, C 11.21 GiB và D 19.34 GiB. Gate hiện chặn tại ngưỡng 2 GiB RAM và 3 GiB commit, thiếu tương ứng khoảng 0.36 và 0.63 GiB tại lần đo đó. Cần đo lại ngay trước mỗi thao tác chính thức.

Snapshot lúc 12:51:41 +07:00 trong khi đang thử build cũng báo `RESOURCE_BLOCKED`, với RAM 1.58 GiB, commit 0.79 GiB, C 11.16 GiB và D 19.28 GiB. [Snapshot CLI](../../outputs/jaes_v2/phase10/empirical_execution_v1/jaes_final_resource_check_v1.json)

Lần đo cuối lúc 12:52:58 +07:00 sau khi dừng build có RAM 1.77 GiB, commit 0.81 GiB, C 11.16 GiB và D 19.28 GiB. Thiếu khoảng 0.23 GiB RAM và 2.19 GiB commit. [Snapshot sau build](../../outputs/jaes_v2/phase10/empirical_execution_v1/jaes_final_resource_after_build_v1.json)

`PRIMARY_1088` là phương án dự kiến sau khi tài nguyên PASS. Primary lấy 256 REAL và 64 FAKE mỗi attack A07–A19, tương ứng 17.408 dự đoán với bốn model và bốn điều kiện nếu đủ support. Fallback 544 / 8.704 chỉ được chọn vì ngân sách trước khi thấy dữ liệu hoặc kết quả, và vẫn cần cùng ngưỡng RAM/commit. Benchmark development gợi ý khoảng 50 phút cho primary và 25 phút cho fallback, chưa gồm ZIP, hash, I/O và bootstrap. Chỉ audit được duyệt mới xác minh support và chênh lệch giữa số trial trong CM protocol và số FLAC trong ZIP.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_experiment resources
```

## Hai phê duyệt độc lập

Prelock SHA-256 `3ae003d298fd812196a3ae0b8e433f8061518805737264fb35cbd57a3465a58f`. Auditor SHA-256 `ea4c425892da335ef8accf7097430eb02cc26adf64153338b95c39ec88c2ac29`. Scoring runner SHA-256 `d36cda8e064c130eea7e7f76c9d048157b900aabd363b02ec412308f953e8a83`. Runner cũ trong prelock là snapshot trước khi tách hai bước. Auditor đóng dấu runner hiện tại khi lập proposed lock. Bốn model xuất, selected-checkpoint seal, contract, score, ba manifest identity và fingerprint engine đã khớp binding trong prelock. [Prelock](../../outputs/jaes_v2/phase10/empirical_execution_v1/heldout_prelock_9107_9207_v1.json)

1. Sau khi tài nguyên PASS, nếu đồng ý đọc official CM protocol và license, đối chiếu archive và tính SHA-256 byte của các FLAC được chọn, tác giả cấp `AUTHOR_APPROVAL_PROTOCOL_READ`. Điền tên, thời điểm, tham chiếu phê duyệt và lý do chọn subset có căn cứ. Chỉ chuyển `allow_selected_audio_byte_hashes` thành true nếu chấp thuận. Lưu bản hoàn chỉnh thành `protocol_read_author_approval.json` trong folder evidence. Mẫu bàn giao có giá trị null/false nên không qua gate. [Mẫu chưa phê duyệt](../../papers/jaes_v2/jaes_manuscript/PROTOCOL_READ_APPROVAL_TEMPLATE.json)
2. Khi có phê duyệt bước 1, chạy lệnh audit bên dưới. Auditor dùng ZIP và metadata, đối chiếu ID/speaker/attack, quota, trùng checksum và license. Audit dừng với proposed lock chưa được phép scoring. Việc hash byte có truy cập nội dung FLAC để tính hash, không giải mã waveform. Không thể xác minh generator lineage nếu official protocol thiếu trường liên kết thích hợp.
3. Rà license, shortlist, shortage, resource và hash của proposed lock. Chỉ sau khi tác giả cấp `AUTHOR_APPROVAL_HELDOUT_SCORING` riêng gắn với proposed lock, selected rows và license mới chạy lệnh scoring ở [hướng dẫn hai cổng](JAES_MANUSCRIPT_AND_HELDOUT_HANDOFF.md). Sprint này chưa tạo bản ghi phê duyệt thực tế, chưa mở official evaluation, chưa train, chưa dùng calibration để chọn phương pháp và chưa chấm điểm.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.heldout_protocol_audit --protocol-approval outputs/jaes_v2/phase10/empirical_execution_v1/protocol_read_author_approval.json --output outputs/jaes_v2/phase10/empirical_execution_v1/heldout_proposed_9107_9207_v1.json
```

## Công việc đến 20/12

10–13/11 đối chiếu source, citations, template và bản thảo. 14–17/11 ưu tiên hai gate tài nguyên và protocol, hoàn thiện build và metadata. 18–25/11 chỉ chấm sau khi có phê duyệt riêng. 26–30/11 phân tích held-out và báo cáo cả kết quả bất lợi. 01–10/12 hoàn tất nội dung, 11–14/12 QA khoa học và LaTeX, 15–20/12 chuẩn bị hồ sơ nộp, disclosure và quyền tái sử dụng.

Bản conference giữ SHA-256 `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073`. Venue ICAIN là liên kết tạm thời đến khi tác giả xác nhận bằng thư chấp nhận. [Ma trận mở rộng và quyền](../../papers/jaes_v2/jaes_manuscript/CONFERENCE_EXTENSION_DISCLOSURE.md) · [Build](../../papers/jaes_v2/jaes_manuscript/README_BUILD.md) · [Audit tài liệu tham khảo](../../papers/jaes_v2/jaes_manuscript/reference_audit.csv)

## Kiểm tra bản thảo

Thân bài có khoảng 5.952 từ theo phép đếm source, bỏ float, trích dẫn và phần lớn công thức. Abstract 138 từ. Kiểm tra tĩnh xác nhận 24 khóa citation duy nhất, đều có tham chiếu, mọi input và hình đều tồn tại, và mọi label tham chiếu được định nghĩa. Bốn hình vector đã được render và kiểm tra bằng mắt. Bảng bổ sung theo điều kiện được tái tính từ CSV đã seal, và inventory đọc kích thước tensor từ export mà không nạp model. 22 test pass trên fixture và development; không có test official.

Đã tích hợp class và style chính thức JAES 2026.08 từ folder tác giả cung cấp. Disclosure ghi OpenAI Codex, nhà cung cấp, ngày và vai trò, với model/version và xác nhận rà soát của tác giả còn chờ. Bản thảo vẫn ngắn hơn đích 7.000–9.000 từ và chưa có kết quả held-out. Không coi hai con số này là lý do để kéo dài bài bằng nội dung lặp.

Tectonic 0.17.0 portable chạy được. Lần build dừng ở khởi tạo format XeTeX khi bundle liên tục tải các gói ngôn ngữ và có timeout mạng. Không cài distribution toàn hệ thống hoặc tải toàn bộ bundle 2,68 GiB. Chưa có PDF, page count, kiểm tra float hoặc log LaTeX hoàn chỉnh. Dùng Overleaf với compiler pdfLaTeX theo README để hoàn tất.

`MANUSCRIPT_SCIENTIFIC_CONTENT_INCOMPLETE` cho gói nộp hiện tại. Phần còn thiếu gồm evidence held-out, build/QA PDF, xác nhận venue và quyền tái sử dụng conference, author metadata và disclosure hoàn chỉnh.
