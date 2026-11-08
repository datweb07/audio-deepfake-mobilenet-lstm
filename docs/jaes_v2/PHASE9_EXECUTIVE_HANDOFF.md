# Phase 9 — Bàn giao đến hạn 20/12/2026

## A. Đã hoàn thành những gì?

Xác minh lại hai seal Phase 8, đo ba đường gradient trên MobileNetV3Small–LSTM, đo thử validation bốn view, triển khai helper guard theo số đếm và tạo manifest medium-plus. Foundation qua preflight kiểm tra dữ liệu, dự báo disk và số update. Bản thảo tiếng Anh dùng số liệu đã có và tách khỏi bản hội nghị. [Master Phase 9](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE9.md)

## B. Những lỗi nào đã sửa?

Nhánh VJP thử nghiệm thay thao tác sửa convolution đang sống bằng một graph chia sẻ biến, xác minh callable gốc vẫn nguyên. Helper guard từ chối nhóm thiếu, denominator lệch hoặc đếm sai. Phase 8 vẫn là bằng chứng lịch sử. Trainer mới và tích hợp lịch validation đang chờ qualification.

## C. Model nào được ưu tiên?

MobileNetV3Small–LSTM, hai baseline ERM/channel ghép theo seed. ShuffleNet là kiểm tra transfer có điều kiện. AASIST/RawNet2 cần audit checkpoint và overlap trước so sánh.

## D. Dataset hiện tại đủ chưa?

Đủ cho qualification và so sánh development. Medium có 640 train, medium-plus 2.560, foundation 25.380. Cần lineage đối chứng cho cue gắn generator và một tập độc lập cho xác nhận bên ngoài.

## E. Có phải tải thêm dataset không?

Chưa. ASVspoof 2021 LA và In-The-Wild vượt dung lượng trống trên mức dự trữ. PhonemeDF mới xác nhận mẫu công khai, còn phải xác minh bản phát hành, quyền sử dụng và ghép nguồn. [Quyết định dữ liệu](PHASE9_DATASET_ACQUISITION_DECISION.md)

## F. Có phương pháp mới chưa?

`NOVELTY_NOT_ESTABLISHED`. Branch B giữ baseline đã biết và chỉ tiến tới phép thử cue khi có lineage. [Đối chiếu prior art](PHASE9_NOVELTY_COMPARISON.md)

## G. Ba contributions hiện đạt mức nào?

C1 có tín hiệu development cho order và lỗi theo nhóm. C2 là giả thuyết chưa xác thực. C3 là hướng selective inference chưa mở calibration.

## H. Có thể full train chưa?

**NO-GO.** Parity mới gồm hai gradient case và bốn validation view. Ổ C còn 536.825.856 byte tại preflight. TensorFlow thấy 0 GPU. Cần nghiệm thu fixed-engine, validation và resource trước khi mở training mới. [Go/no-go](PHASE9_FULL_TRAINING_GO_NO_GO.md)

## I. Người dùng nên chạy lệnh gì?

Hai run đã được kiểm tra lại. Khi cần tái lập audit, chạy lệnh đã kiểm tra trong gói lệnh. Sau khi có headroom trên C, ưu tiên qualification pilot 100 update có test resume trước mọi lượt replication. [Gói lệnh](PHASE9_AUTHOR_TRAINING_COMMANDS.md)

## J. Nguy cơ không kịp 20/12?

Cao nếu đặt điều kiện phải có C2, external confirmation và full training cùng lúc. Đặt mốc quyết định theo evidence. Nếu không qua gate, hoàn thiện báo cáo development với giới hạn rõ ràng và đánh giá lại loại bài cũng như thời điểm nộp. [Lịch và quyết định](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE9.md)
