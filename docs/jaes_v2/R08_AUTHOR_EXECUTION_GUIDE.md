# R08 — hướng dẫn tác giả chạy thủ công

Hiện chỉ được chạy `resources`, `integrity`, `metadata` và `replay`. Metadata/replay đã có kết quả write-once. Đọc JSON đã lưu để xem kết quả. R08 dùng frozen inference và không có bước train.

```powershell
Set-Location D:\audio-deepfake-mobilenet-lstm
$py = '.\.venv\Scripts\python.exe'
& $py -B -X utf8 -m scripts.jaes_v2.r08_external resources
& $py -B -X utf8 -m scripts.jaes_v2.r08_external integrity
```

## Sau khi quyền sử dụng được làm rõ

Lưu phản hồi thật vào `outputs/jaes_v2/r08_external_v1/data_use_confirmation.md`. Đọc ba câu hỏi cần xác nhận trong báo cáo quyền. Sao chép [mẫu có sẵn](../../outputs/jaes_v2/r08_external_v1/author_data_access_approval.template.json) thành `author_data_access_approval.json` trong cùng thư mục, rồi điền thông tin thực. Hai hash lấy bằng `Get-FileHash -Algorithm SHA256`, chuyển sang chữ thường. Nếu có đối chiếu alias, lưu `speaker_aliases.json` dạng ánh xạ tên chuẩn hoá sang tên chuẩn hoá và điền hash. Giữ null khi không có mapping. Ngân sách 256 MiB dưới đây là ví dụ có giới hạn cho Range. Kiểm tra mục tài nguyên trước khi chạy.

```json
{
  "scope": "AUTHOR_APPROVAL_R08_DATA_ACCESS",
  "dataset": "mueller91/In-The-Wild",
  "revision": "eee168f92c367f8c82ff2cf42b6f61e362fd6211",
  "intended_use": "LOCAL_RESEARCH_AND_AGGREGATE_REPORTING",
  "destination": "outputs/jaes_v2/r08_external_v1/audio",
  "rights_sha256": null,
  "confirmation_sha256": null,
  "speaker_aliases_sha256": null,
  "method": "range",
  "max_download_bytes": 268435456,
  "author": null,
  "approved_at": null,
  "approval_reference": null
}
```

`rights_sha256` là hash [báo cáo quyền](R08_DATA_RIGHTS_AND_PROVENANCE.md). `confirmation_sha256` là hash thư xác nhận thật. Ghi timestamp ISO 8601 có offset. `approval_reference` dẫn tới quyết định thực của tác giả. Chạy tuần tự, chỉ chuyển bước khi bước trước đạt:

```powershell
& $py -B -X utf8 -m scripts.jaes_v2.r08_external pilot
& $py -B -X utf8 -m scripts.jaes_v2.r08_external acquire
& $py -B -X utf8 -m scripts.jaes_v2.r08_external lock
& $py -B -X utf8 -m scripts.jaes_v2.r08_external scoring-template
```

Pilot 32 clip chỉ kiểm decode. Nếu phải dừng giữa pilot/acquire, giữ WAV đã có và chạy lại cùng lệnh, cùng approval và cùng thiết kế. Không đổi alias, byte cap hoặc subset giữa một lần chạy dở. Nếu lỗi CRC, decode, hết byte cap hoặc quota thiếu sau audit, giữ bằng chứng và rà trước khi tạo thiết kế mới.

Nếu HTTP Range không dùng được, chỉ chuyển sang phương án ZIP **sau khi xác nhận riêng** và rà lại kế hoạch trước khi tiếp tục. Approval mới phải ghi `method: local-zip` và cap ít nhất 8.161.489.023 bytes. Chạy `zip-preflight`, rồi lưu archive chính thức đã ghim tại `outputs/jaes_v2/r08_external_v1/release_in_the_wild.zip`. CLI kiểm kích thước và SHA toàn file trong pilot/acquire. Giữ một bản ZIP, chỉ đọc WAV cần chọn. Phương án này không tự tải ZIP và không tự thay approval của pilot đã có.

## Sau khi có protocol và phê duyệt scoring riêng

Rà `configs/jaes_v2/r08_external_v1/protocol.json`, `subset.json`, scope cross-dataset diagnostic, danh sách model, số dự đoán và các giới hạn lineage. Sao chép `author_scoring_approval.template.json` thành `author_scoring_approval.json`. Điền đúng tên, timestamp có múi giờ và tham chiếu xác nhận thật. Giữ nguyên mọi hash. Chỉ tác giả chạy:

```powershell
& $py -B -X utf8 -m scripts.jaes_v2.r08_external score
& $py -B -X utf8 -m scripts.jaes_v2.r08_external verify
& $py -B -X utf8 -m scripts.jaes_v2.r08_external integrity
```

Nếu score bị ngắt, dùng lại `score` với chính protocol và approval đó. Các shard hợp lệ được bỏ qua. Sau seal dùng `verify` để phát lại báo cáo. Báo cáo cuối chỉ được tạo khi đủ lưới dự đoán. Với lineage còn thiếu, công bố hiệu ứng theo seed và đếm lỗi; khoảng tin cậy speaker chưa đủ căn cứ. V8 chỉ xem xét sau khi rà kết quả thực và phạm vi đã thống nhất với giảng viên.

## Bàn giao

Gửi báo cáo quyền, báo cáo acquisition, hướng dẫn này và `metadata.json`, `resources.json`, `inference_qualification.json`, `protected_after.json`. Nếu cần rà code, gửi hai module R08 và test tương ứng. Sau khi khóa, gửi protocol và báo cáo audit đã lược thông tin phê duyệt cá nhân. Sau khi chấm, gửi `run/metrics.json`, `run/seal.json` và báo cáo kết quả. Kiểm tra điều khoản trước khi chia sẻ score cấp bản ghi hoặc manifest chứa tên speaker. Giữ WAV, ZIP và văn bản phê duyệt trong repo local.
