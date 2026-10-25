# Prompt nâng cấp nghiên cứu LAVA cho JAES

Ngày soạn: 24/10/2026. Sao chép phần từ “BẮT ĐẦU PROMPT” đến hết để giao cho AI có quyền đọc repository và tìm tài liệu trên web. Đây là đề bài nghiên cứu và triển khai, không phải chứng nhận tính mới hoặc cam kết bài sẽ được nhận.

Cập nhật mốc bài accept ngày 24/10/2026 theo xác nhận mới nhất của tác giả: `papers/PaperID_804_final.docx` (tên có dấu gạch dưới). Đã đọc nội dung, cả năm bảng và các công thức OMML trong file này, rồi đối chiếu với file tên có khoảng trắng. Số liệu bảng không thay đổi; có khác biệt về diễn đạt, trình bày đóng góp, thứ tự và nội dung bibliographic references. Bản có dấu gạch dưới là nguồn chuẩn duy nhất cho việc đối chiếu phần mở rộng.

## BẮT ĐẦU PROMPT

Bạn đóng vai trò cộng tác viên nghiên cứu audio anti-spoofing, kỹ sư ML và người phản biện. Hãy trực tiếp nâng cấp repository `D:\audio-deepfake-mobilenet-lstm` thành một nghiên cứu bản mở rộng có thể được đánh giá cho Journal of the Audio Engineering Society (JAES). Làm việc dựa trên bằng chứng; không dừng ở đề xuất hoặc scaffolding nếu vẫn còn công việc có thể thực hiện trong môi trường.

Mục tiêu là kiểm chứng ít nhất BA ĐÓNG GÓP mới so với bài đã accept và có khác biệt thực chất với nghiên cứu quốc tế gần nhất. Không mặc định ba ý tưởng dưới đây là mới. Nếu không tìm được khác biệt hoặc thí nghiệm bác bỏ giả thuyết, báo rõ, sửa hướng và cập nhật giới hạn; không đổi tên kỹ thuật cũ để đủ số đóng góp.

### 1. Mốc bài gốc và quy tắc bảo toàn

Bản đã accept mà tác giả xác nhận mới nhất là `papers/PaperID_804_final.docx`, tên “LAVA: A Lightweight Benchmarking Framework for Robust and Efficient Deepfake Voice Detection”. Đường dẫn đầy đủ: `D:\audio-deepfake-mobilenet-lstm\papers\PaperID_804_final.docx`. SHA-256 tại thời điểm đọc: `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073`. Tác giả cho biết bài đã review và accept, chưa phát hành và chưa thuyết trình. Tên hội nghị, nhà xuất bản, điều khoản bản quyền và DOI chưa được cung cấp. Không suy diễn địa điểm công bố từ PaperID hoặc định dạng Word.

Đọc file Word này trước, bao gồm bảng, công thức và references; sau đó đọc README, config.py, AUDIT_REPORT.md, docs/SOURCE_PROVENANCE.md, registry, data manifests, training policies, stress runners, metric code, tests và các audit dưới papers. File tên có khoảng trắng `papers/PaperID 804 final.docx`, các bản ok.tex và LAVA_FULL_PAPER* là tài liệu phụ; không thay thế bản Word có dấu gạch dưới làm mốc novelty. Nếu file chuẩn không còn hoặc hash thay đổi, ghi rõ và xác minh bản mới trước khi dùng các con số tóm tắt bên dưới; không tự động fallback sang file gần giống tên.

Ba đóng góp ĐÃ CÓ trong bản accept, phải ghi nguyên ý vào extension matrix và không tính lại thành đóng góp mới:

1. Clean evaluation contract có checksum-group-disjointness, artifact traceability, ROC-AUC/EER, score semantics và công khai provenance.
2. Matched diagnostics trên fixed subset cho các mức noise, codec settings và simulated replay, kèm aggregate failure-pattern analysis.
3. Timing trên desktop CPU một luồng và exploratory Pareto ba mục tiêu để chọn artifact theo scenario.

Các RQ cũ tương ứng là so sánh deployed artifacts trên clean data trong điều kiện training/calibration khác nhau; biến động absolute performance trên diagnostic subset; và non-dominance theo diagnostic EER, mean stressed F1, E2E RTF. Hạ tầng provenance, stress diagnostics và Pareto là nền kế thừa. C1/C2/C3 của journal phải bổ sung câu hỏi/phương pháp/phát hiện vượt ba đóng góp này, không chỉ đổi tên hoặc tăng số model/conditions.

Các thông tin sau được trích từ bản Word để định hướng, phải tự đối chiếu:

- 18.722 file quét, 18.232 file giữ lại; train/validation/test = 12.762/2.733/2.737; seed 42.
- Chỉ xác lập checksum-group disjointness; speaker/source/generator/parent-recording/corpus chưa có metadata đáng tin cậy.
- Bốn mô hình Mel-sequence gồm MobileNetV3Small, ShuffleNetV2, MnasNet-A1, EfficientNet-B0, sau backbone là LSTM; RawNet2/AASIST là artifact pretrained bên ngoài.
- ShuffleNet có clean F1 0,9824, ROC-AUC 0,9929, EER 1,46%. MobileNet có E2E latency 43,81 ms trên một CPU desktop.
- Bảng 3 còn có PR-AUC 0,9898 và specificity 0,9848 cho ShuffleNet. Clean F1 của ShuffleNet là 0,9824 trên 2.737 recordings, nhưng diagnostic-subset clean F1 là 0,9756 trên 100 recordings; không trộn hai cohort.
- Robustness chỉ trên 100 recordings, gồm AWGN, codec round trip và replay mô phỏng. Bản accept nói không giữ paired clean/stressed scores để phân tích prediction flips; phải kiểm kê artifact hiện tại trước khi kết luận còn/thiếu evidence.
- Bản Word dùng Pareto MIN diagnostic EER, MAX mean stressed F1, MIN E2E RTF; frontier là MobileNet và ShuffleNet. Không lấy kết luận AASIST nằm trên frontier từ bản cũ khác.
- Công thức vector mục tiêu trong bản accept là `g(m) = (EER_diag(m), -mean_F1_stress(m), RTF(m))`; mean stressed F1 là trung bình của chín điều kiện. ShuffleNet có mean stressed F1 0,8133 và RTF 0,0208; MobileNet có RTF 0,0146. Giữ đúng dấu và phạm vi cohort khi tái hiện.
- Bản accept báo exact McNemar với Holm correction cho fixed-test predictions; adjusted p = 0,0096 khi so ShuffleNet với MobileNet. Đây không phải bằng chứng khác biệt qua nhiều training seeds và không phải một đóng góp thống kê mới của journal.
- Bảng 2 ghi EfficientNet là warm-up checkpoint, fine-tuning chưa hoàn tất; local thresholds lần lượt MobileNet 0,82, ShuffleNet 0,12, MnasNet 0,90, EfficientNet 0,90, còn external references dùng mặc định 0,50. Threshold selection theo validation F1 không đồng nghĩa probability calibration.
- Không có bằng chứng trong bài gốc về unseen-corpus, physical replay, causal streaming, edge device hoặc multi-seed.
- Training và calibration giữa các artifact không đồng nhất. Không được dùng EER gần ngẫu nhiên của external checkpoints để chứng minh kiến trúc đề xuất vượt RawNet2/AASIST nói chung.

Giữ nguyên bản Word, checkpoint và benchmark lịch sử. Kiểm tra git status, giữ thay đổi sẵn có của tác giả; tạo nhánh codex/jaes-research-upgrade nếu phù hợp quyền môi trường. Dùng output namespace riêng `outputs/jaes_v2/`, config và artifact version mới. Không reset hoặc xóa dữ liệu. Không tự nộp bài, gửi email hay tải dữ liệu riêng lên dịch vụ ngoài.

### 2. Xác minh journal và tính hợp lệ của bản mở rộng

Đọc trực tiếp các nguồn chính thức và ghi URL/ngày truy cập vào `docs/jaes_v2/JOURNAL_CHECK.md`:

- https://aes.org/publications/journal/
- https://aes.org/publications/journal-author-guidelines/
- https://aes.org/publications/aes-journal-manuscript-types/
- https://aes.org/publications/submit-a-paper-to-the-aes-journal/
- https://mjl.clarivate.com/home — tìm đúng title và ISSN 1549-4950.

Trang https://wos-journal.info/journalid/395 là nguồn độc lập, không phải Clarivate. Khi không mở được hồ sơ MJL hoặc không có JCR, ghi `NOT_VERIFIED` cho indexing/JIF/quartile tương ứng. Không suy ra quartile chỉ từ JIF hay một percentage rank thiếu năm và category. Tách “journal đang hoạt động/nhận bài” khỏi “SCIE được xác minh chính thức” và “đơn vị của tác giả công nhận”.

Kiểm tra chính sách conference extension, prior publication, quyền tái sử dụng bảng/hình, AI disclosure và template hiện hành. Không tự đặt quy tắc 30% hoặc 50% nội dung mới nếu AES không nêu. Việc bài gốc chưa phát hành không làm bài accept trở thành nghiên cứu chưa từng công bố theo mọi nghĩa; phải khai báo trạng thái thực tế. Nếu bài gốc là journal paper, không mặc định áp dụng ngoại lệ conference extension. Hỏi tác giả đúng phần còn thiếu về venue/copyright khi cần quyết định điều kiện nộp; tiếp tục audit và triển khai độc lập trong lúc chờ.

Lập `EXTENSION_MATRIX.md`: từng đóng góp, phương pháp, dữ liệu, thí nghiệm, bảng/hình của bài gốc; phần tái sử dụng và phần mới; bằng chứng; yêu cầu trích dẫn/quyền sử dụng. Chuẩn bị cover letter dạng DRAFT với thông tin chưa biết đánh dấu rõ. Không xác nhận eligibility tuyệt đối trước khi biết venue và điều khoản liên quan.

Extension matrix phải có tối thiểu ba hàng đối chiếu trực tiếp: (i) old clean/provenance contract → C1 protocol và phát hiện mới về held-out generator/channel compositions; (ii) old fixed-subset aggregate stress diagnostics → paired full-cohort evidence và C2 learning method; (iii) old static desktop Pareto selection → C3 inference policy với measured risk/coverage/cost. Chỉ mở rộng coverage hoặc sửa thiếu sót của bài cũ phải được gắn nhãn validation/infrastructure, không tự tính là một contribution độc lập. Nêu rõ phần nào không có trong bài accept và phần nào là kết quả mới đã thực nghiệm.

### 3. Rà soát tính mới trên tài liệu quốc tế

Tìm tài liệu gốc đến ngày thực hiện, ưu tiên AES E-Library/JAES, ISCA Archive, ICASSP, ASVspoof, nhà xuất bản, PMLR/OpenReview, arXiv và official repos. Phân biệt paper peer-reviewed với preprint. Search snippet chỉ dùng để tìm nguồn, không làm bằng chứng chi tiết.

Khởi đầu tối thiểu từ các hướng sau và xác minh nội dung thực tế:

- Benchmarking Audio Deepfake Detection Robustness in Real-world Communication Scenarios, EUSIPCO 2025: https://eusipco2025.org/wp-content/uploads/pdfs/0000566.pdf
- Towards Generalisable and Calibrated Audio Deepfake Detection with Self-Supervised Representations, Interspeech 2024: https://doi.org/10.21437/Interspeech.2024-1302
- Replay Attacks Against Audio Deepfake Detection, Interspeech 2025: https://doi.org/10.21437/Interspeech.2025-20
- Scaling Behavior in Model Fine-tuning for Audio DeepFake Detection, ICML 2026: https://proceedings.mlr.press/v306/li26u.html
- Các công trình RawBoost, SSL anti-spoofing, cross-domain/multilingual benchmarking, CodecFake, knowledge distillation, group distributionally robust optimization, consistency learning, calibration, selective classification và adaptive inference.

Tìm cả nghiên cứu uncertainty/conformal/evidential anti-spoofing, teacher-student learning trên compressed speech, channel composition và watermark shortcuts; không coi các từ khóa này tự thân là mới.

Đọc references từ chính `PaperID_804_final.docx`, không kế thừa numbering của file tên có khoảng trắng: non-semantic representations là [15], replay attacks [16], XMAD-Bench [17], MobileNetV3 [18], ShuffleNetV2 [20], MnasNet [22], EfficientNet [23] và librosa [24]. Khi viết journal manuscript, đánh số lại theo thứ tự trích dẫn mới. Bản chuẩn có thay đổi DOI/reference metadata so với bản đọc trước, ví dụ reference [2] hiện ghi `10.1145/3714458`; đây là dữ liệu trích từ file, chưa tự động là chứng nhận reference đúng toàn bộ. Xác minh title/authors/year/venue/DOI từ nguồn gốc; phân biệt arXiv DOI với DOI bản proceedings và ưu tiên bản xuất bản chính thức phù hợp. Không sao chép lỗi định dạng hoặc thiếu author/year từ danh mục cũ.

Tạo `LITERATURE_MATRIX.csv` với tối thiểu 20 công trình liên quan trực tiếp nếu đủ nguồn xác minh. Mỗi dòng có title, năm, DOI/URL, publication status, code URL, training data, splits, channel conditions, method, baselines, latency hardware, limitation, liên hệ với C1/C2/C3 và mức độ đã đọc. Không chèn tài liệu ít liên quan để đủ số. Chọn 5–8 nearest neighbors để so sánh sâu. Tìm bài JAES liên quan thật sự và giải thích scope; không nhét citation để lấy lòng journal.

Tạo `NOVELTY_GATE.md`. Mỗi contribution cần: vấn đề cụ thể, prior art gần nhất, khác biệt kỹ thuật chính xác, giả thuyết có thể bác bỏ, thí nghiệm phân biệt với giải pháp cũ, điều kiện thất bại và mức tự tin. Trạng thái ban đầu phải là `PROPOSED`, chỉ nâng thành `SUPPORTED` khi cả literature và kết quả cho phép. Nếu không chứng minh được ba đóng góp, báo không đạt mục tiêu và đề xuất hướng tiếp theo; không bịa claim “first”, “world-first” hoặc SOTA.

### 4. Câu hỏi nghiên cứu thống nhất

Mục tiêu trung tâm: một bộ phát hiện giọng nói giả nhẹ có giữ được khả năng phân biệt và độ tin cậy khi gặp cả generator chưa thấy và chuỗi xử lý âm thanh chưa thấy, dưới ngân sách suy luận cụ thể hay không?

Ưu tiên nghiên cứu cơ chế tác động của chuỗi codec/noise/reverberation lên dấu vết giả mạo, học biểu diễn bền vững, rồi ra quyết định có quyền từ chối khi không đủ tin cậy. Không mở rộng vô hạn sang video, LLM giải thích, watermark generation hay hàng chục backbone không phục vụ giả thuyết.

Ba nhóm dưới đây là ứng viên đóng góp. Có thể sửa hoặc thay sau novelty gate, nhưng phải giữ logic khoa học thống nhất và lập luận vì sao hướng mới tốt hơn.

### 5. C1 — Protocol và phát hiện thực nghiệm về đồng thời hai loại dịch chuyển

Giả thuyết H1: generalization sang generator mới và robustness với channel mới không được dự đoán đầy đủ từ clean EER; tổ hợp và thứ tự các phép xử lý tạo failure modes vượt hiệu ứng từng phép riêng lẻ.

Xây protocol có nguồn gốc đầy đủ cho generator-held-out, corpus-held-out và channel-composition-held-out. Tách rõ unknown generator, unknown corpus, unknown codec/bitrate và unseen chain; không gọi chung là OOD khi chưa định nghĩa. Một test cohort có thể đo đồng thời nhiều dịch chuyển, nhưng muốn tách hiệu ứng phải có factorial cells phù hợp. Dữ liệu không đủ để tách thì giới hạn suy luận.

Yêu cầu dữ liệu:

- Dữ liệu LAVA cũ chỉ là legacy comparison, không làm bằng chứng duy nhất cho generalization.
- Chọn dữ liệu public có protocol/metadata rõ, ví dụ ASVspoof official train/dev và các eval phù hợp; chọn thêm ít nhất một corpus ngoài họ dữ liệu dùng huấn luyện nếu license, tải xuống và compute cho phép. Kiểm tra lineage trước khi gọi ASVspoof 2019→2021 là hoàn toàn độc lập.
- Lưu dataset_id/version/license/download origin/speaker_id/generator_id/attack_id/language/parent_id/source_id; không suy diễn UNKNOWN thành nhãn chắc chắn. Với voice conversion, xem cả source và target speaker nếu metadata có.
- Mỗi miền phải có REAL và FAKE thích hợp. Tránh corpus A chỉ REAL và corpus B chỉ FAKE khiến mô hình học dataset fingerprint. Đo metadata-only baseline, duration/sample-rate/silence baseline để phát hiện shortcut.
- Không tải hàng trăm GB trước khi kiểm kê dung lượng, license, thời gian và đường dẫn. Khi thiếu dữ liệu, tạo importer, validation, download instructions và trạng thái `MISSING_DATA`, không tạo mẫu giả thay kết quả thật.

Ràng buộc split:

- Respect official partitions cho kết quả official; protocol tùy chỉnh là stratum riêng, không gắn nhãn official leaderboard.
- Group toàn bộ crop, transcoded variants và channel variants theo parent recording. Hold-out theo generator/speaker khi có metadata xác thực; công bố giao cắt còn lại.
- Kiểm tra exact bytes, decoded-waveform duplicates và near duplicates theo phương pháp có threshold đánh giá. Fingerprint/embedding similarity dùng để gắn cờ rồi review; không khẳng định chứng minh speaker independence.
- Corpus audit bao gồm pretraining provenance/possible overlap của teacher và baseline. Không biết thì ghi uncertainty, không bảo đảm “unseen” vượt phần đã xác minh.
- Validation được chia thành model selection và calibration/policy selection không giao nhau theo group. Final tests niêm phong; không dùng test errors để chọn hyperparameter, perturbation range hoặc policy. Nếu đã xem test để thiết kế v2, coi đó là development và cần untouched confirmation set.

Stress conditions: clean, environmental noises có license, AWGN kiểm soát, measured RIR hoặc simulator ghi đúng tên, codec/bitrate hợp lệ, resampling và chuỗi 2–3 phép. Chọn ma trận nhỏ có căn cứ audio engineering trước; không cần full combinatorial grid. Có codec→noise so với noise→codec và codec→reverb so với reverb→codec khi hợp lý vật lý; nêu noise nằm ở giai đoạn acoustic capture hay post-decode.

Mỗi phép biến đổi phải lưu source_id/parent_id, thứ tự, random seed, actual sample rate/length, SNR định nghĩa trên đoạn nào, peak/clipping, codec delay/padding, container và FFmpeg/build/version, RIR ID, transform hash. Ghi chính sách alignment và time trimming; không để độ dài hay encoding header trở thành nhãn giả mạo. Áp dụng phân bố channel tương đương cho cả hai lớp.

Lưu paired clean/stressed raw scores cho toàn bộ evaluated cohort theo sample_id, model_id, seed, checkpoint và condition_id. Report clean→error flips, real false alarms và fake misses riêng, worst group, per-generator/per-corpus/per-condition. Bootstrap phải resample theo independent parent/speaker khi phù hợp, giữ các variants cùng group.

Định nghĩa interaction trên loss/risk hoặc metric phù hợp trước phân tích; không khẳng định hai phần trăm EER cộng tuyến tính theo mặc định. Xác định R_AB, R_A, R_B, R_clean, interaction contrast và order contrast kèm CI; nêu tính mô tả và giới hạn causal inference. C1 chỉ trở thành đóng góp nghiên cứu khi tạo phát hiện không tầm thường, được kiểm chứng trên held-out data và có giá trị ngoài dataset riêng. Việc thêm vài codec hoặc SHA256 đơn thuần là hạ tầng.

### 6. C2 — Student nhẹ học theo cặp để bền vững với chuỗi xử lý

Giả thuyết H2: học từ paired clean/channel views và lỗi worst-group có thể giữ forensic cues tốt hơn ERM/random augmentation/standard KD ở generator và channel chưa thấy, tại cùng inference budget.

Dùng MobileNet hoặc ShuffleNet hiện có làm student khởi đầu. Không thay đồng loạt mọi backbone. Thêm teacher SSL chống spoof mạnh, có official implementation và provenance xác minh; teacher chỉ dùng trong training. Baseline không có teacher cũng phải đủ mạnh và được train hợp lý. Không distill từ hai external checkpoints gần chance trên LAVA mà chưa kiểm chứng chất lượng.

Thiết kế công thức rõ trước khi code. Một khung ứng viên để khảo sát:

L = L_sup + lambda_kd * w(x,c) * T^2 * KL(p_teacher(x)/T || p_student(c(x))/T) + lambda_pair * D(z_student(x), z_student(c(x))) + lambda_group * L_group.

Viết đúng softmax/sigmoid temperature trong triển khai, thay ký hiệu rút gọn trên bằng định nghĩa toán học chuẩn. Teacher dừng gradient; projection dimensions, normalization, distance D, group weights/update và minibatch sampling phải được mô tả đầy đủ. L_group phải xác định cụ thể nếu dùng, tránh đếm L_sup hai lần mà không giải thích. Teacher reliability weight w phải học/ước lượng từ training/development hợp lệ, có bounded range; teacher confidence tự thân không chứng minh reliability. So sánh w=1, confidence weighting chuẩn và cơ chế đề xuất. Nếu teacher sai có hệ thống trên một miền, đo tác hại thay vì che bằng weighting.

Không coi công thức tổ hợp này đã mới. Phải tìm các paired consistency, domain generalization, robust distillation và group-DRO tương tự. Xác định thành phần nào là khác biệt kỹ thuật có ý nghĩa, tại sao giúp giữ forensic cue và thí nghiệm nào có thể bác bỏ giải thích đó. Nếu chỉ là tổ hợp quen thuộc mà không có phát hiện đáng kể, thay hướng hoặc hạ claim.

Ablation tối thiểu cùng backbone/data/tuning budget:

1. ERM clean.
2. ERM + standard augmentation/RawBoost phù hợp và được xác minh.
3. Standard teacher-student KD.
4. Paired consistency không teacher.
5. Group-robust training tiêu chuẩn nếu dùng thành phần này.
6. Full method, cùng các biến thể bỏ từng thành phần mới.

Các baseline phải có cùng augmentation/data access khi kiểm tra đóng góp objective; thêm baseline dùng đúng cùng transformed views nhưng objective tiêu chuẩn. Tách gain từ extra data, teacher, regularization và training duration. Report teacher upper reference, training cost, student inference cost; không tính teacher vào inference nếu thực tế không dùng và không giấu chi phí training.

Khảo sát ảnh hưởng Mel-RGB resize/normalization, sample rate và truncation so với log-Mel 1-channel hoặc LFCC ở phạm vi có lý do. Không đưa đồng thời mọi feature/attention vào model. So sánh temporal pooling với LSTM để xác định giá trị temporal modeling; nếu đề xuất causal/streaming thì loại look-ahead và chứng minh state handling, không lấy offline RTF làm chứng nhận.

Evaluation khóa trước: macro EER across held-out groups, worst-group risk, real false alarms tại operating point chọn trên development, clean quality retention và latency/model-size budget. Gains cần lặp lại trên ít nhất hai external test contexts có provenance được kiểm tra; không cherry-pick corpus thuận lợi. Nếu compute thiếu, ghi contribution mới ở mức pilot.

### 7. C3 — Policy suy luận có từ chối dưới ngân sách tài nguyên

Giả thuyết H3: policy dùng thêm tín hiệu bất đồng giữa các đoạn và độ ổn định theo channel có thể phân bổ chi phí suy luận và từ chối mẫu khó tốt hơn confidence threshold thông thường, ở cùng coverage và measured resource budget.

Chọn thiết kế tối giản sau C1/C2, ví dụ stage 1 student nhẹ, stage 2 thêm temporal context hoặc detector thứ hai chỉ khi cần, và output REAL/FAKE/ABSTAIN. Không bắt buộc heavy teacher vào deployment. Nếu policy dùng hai mô hình hoặc test-time transforms, tính đủ loading, preprocessing, memory và extra inference. Không dùng generator/corpus ground truth để route test sample. Nếu dùng estimated quality/channel features, ghi rõ cách đo, độ trễ và dependency.

Formulate policy objective rõ, ví dụ minimize development selective risk với constraints coverage >= c_min, expected cost <= B và real false alarm <= a_max khi có đủ data. Dùng mục tiêu/constraints hoặc Lagrangian nhất quán; định nghĩa đơn vị cost và infeasible handling. Chọn budgets theo workload và thiết bị đo, khóa trước final evaluation. Constraints trên dev không phải guarantee trên shifted test.

So sánh với single-stage confidence abstention, entropy/max-probability, temperature-scaled threshold, fixed ensemble, random routing có matched expected cost, cùng cascade/routing baseline gần nhất tìm được. Không thể chỉ so method có quyền abstain với baseline forced binary rồi kết luận accuracy tăng. Report matched coverage/cost, forced-decision quality, overall utility gồm abstention và error rates trên các nhóm bị từ chối.

Calibration dùng proper scoring rules: NLL, Brier và reliability/ECE với binning công bố. Chỉ gọi sigmoid output là score P(FAKE) theo contract; chưa có calibration evidence thì không gọi confidence 90% là xác suất đúng 90%. Temperature scaling fit trên calibration set riêng. Policy tuning và đánh giá calibration cần split/cross-fit hợp lệ để tránh dùng cùng dữ liệu làm certificate và chọn policy.

Nếu dùng conformal/risk control, phải nêu calibration unit, exchangeability, finite-sample quantile, loại guarantee (marginal/class-conditional/selective risk), và proof hoặc citation đúng. Label-conditional conformal không mặc định kiểm soát false alarm trên accepted predictions. Không mở rộng guarantee IID sang arbitrary domain shift. Có thể chỉ dùng abstention empirical và đo coverage failures dưới shift; không cần thêm định lý giả để bài trông mạnh.

Đo risk–coverage/AURC, coverage theo lớp/corpus/generator, false alarm và miss rates, stage-2 trigger rate, abstention rate, latency P50/P95, expected compute và worst-case latency. C3 chỉ được gọi mới khi vượt standard routing/calibration với cơ chế khác biệt được novelty gate và ablation hỗ trợ. Quantization hoặc temperature scaling riêng lẻ không phải contribution mới.

### 8. Thiết kế thí nghiệm đủ để review

Tạo `EXPERIMENT_PLAN.md` và configs khóa trước khi chạy final tests. Tách development sweeps, confirmatory runs và exploratory follow-ups. Mỗi experiment có hypothesis, dataset/split hashes, model/seed, selection rule, metric, cost estimate, expected artifact và completion criteria. Không thay đổi plan sau test mà không version lại.

- Legacy stratum: tái hiện số của bài accept từ đúng artifact nếu còn đầy đủ; khác biệt phải audit, không ép tái hiện bằng sửa số.
- Controlled stratum: train các baseline chính và proposed student trên cùng protocol, duration/data access/pretraining disclosure và tuning budget hợp lý. Official-pretrained transfer stratum báo riêng. Có ít nhất một SSL anti-spoofing baseline mạnh ngoài RawNet2/AASIST, dựa trên nguồn hiện hành.
- References giữ native preprocessing hợp lệ; report cả native contract và matched-duration khi cần. Không ép một contract làm hỏng reference rồi gọi so sánh công bằng. Hparam khác nhau được phép khi có tuning budget tương đương và minh bạch.
- Production runs tối thiểu 3 independent training seeds cho method và baselines then chốt; ưu tiên 5 nếu variance/compute cho phép. Một seed + bootstrap không thay được repeated training. Dùng paired seed schedules khi thiết kế phù hợp, không trình bày ba seed như ba dataset độc lập.
- Report seed-level mean/std và paired effect-size CI; bootstrap phân cấp theo seed/group nếu phù hợp. Tránh pseudoreplication từ nhiều variants của cùng utterance. Khóa primary comparisons và correction cho multiple testing; không yêu cầu mọi p-value nhỏ mới công bố.
- Metrics chính: EER, ROC-AUC, PR-AUC, FPR/FNR tại threshold chọn từ dev; t-DCF/minDCF chỉ khi có protocol, costs/priors và ASV scores cần thiết. EER trên test là ranking diagnostic, không phải threshold deployment được học từ test.
- Không gộp F1 khác prevalence thành một mean không giải thích; report per-domain, macro aggregation và sample counts. Worst-group metric phải có denominator/CI và tránh suy luận từ group quá nhỏ.
- Full cohort stresses cho kết luận full-test; subset có lý do phải ghi subset. Paired clean/stress outputs phải đầy đủ và join validated 1:1.
- Một subset robustness cũ đã được xem không làm final confirmation set mới.

Dựng ma trận tối thiểu khả thi trước, rồi chỉ mở rộng khi trả lời giả thuyết mới hoặc failure. Đề xuất ba mức tài nguyên: CPU smoke/pilot, một GPU nghiên cứu khả thi, và confirmatory mở rộng. Kiểm kê hardware thật và dry-run để ước lượng; không bịa GPU-hours. Không tự thuê GPU hoặc phát sinh chi phí. Nếu môi trường thiếu, hoàn tất code/config/test nhỏ, xuất launch scripts và ghi `AWAITING_COMPUTE`; không báo nghiên cứu đã hoàn thành.

### 9. Đo hiệu năng triển khai

Giữ quy ước timing rõ: decode/resample/features, model-only, calibration/routing và E2E. Batch 1, warm-up/repeats, backend thread count, precision, execution provider, hardware/OS/runtime versions, file I/O policy và timer/synchronization đều được seal. Đo trên nhiều clip/duration thay vì lặp một clip rồi suy rộng.

Desktop là baseline deployment environment. Chỉ claim edge khi đo trên thiết bị edge thật; thiếu thiết bị thì `NOT_MEASURED`. INT8/PTQ/QAT là tùy chọn sau method validity; representative calibration data chỉ từ training/development hợp lệ. Kiểm tra conversion fidelity, post-quantization calibration/EER/false alarms trên mọi condition chính, không chỉ file size. RSS toàn process và tensor/activation memory cần phân biệt. Energy chỉ report khi có measurement method/instrument và uncertainty; không suy watts từ latency.

Nếu không triển khai causal streaming thật, dùng ngôn ngữ faster-than-duration offline inference. Nếu có, phải đo acquisition window, look-ahead, hop, state resets, first-decision latency và final-decision latency; clip 3 s mất 40 ms xử lý vẫn phải đợi audio thu vào.

### 10. Kiến trúc source code và artifact contract

Tái sử dụng registry, P(FAKE) semantics, artifact checks và worker isolation hiện có. Không trộn TensorFlow/PyTorch vào một process nếu gây phá environment hoặc sai memory measurement. Không tự upgrade major dependencies trước compatibility audit. Pin dependencies khi cần và có migration note.

Đề xuất cấu trúc sau, được phép điều chỉnh để phù hợp repository:

```text
configs/jaes_v2/                 # schema, dataset/split, model, train, stress, policy, eval
src/lava/data/                  # importers, provenance, group split, overlap audit
src/lava/robustness/            # ordered transforms, cache, deterministic replay
src/lava/training/              # paired sampler, distillation, robust objectives
src/lava/calibration/           # proper calibration, policy fitting, abstention
src/lava/models/                # new student/adapters preserving existing IDs
benchmark/jaes_v2/              # paired evaluation, group reports, deployment runs
scripts/jaes_v2/                # preflight, dry-run, launch/resume, reproduce
tests/                         # meaningful leakage, math, lifecycle, integration tests
outputs/jaes_v2/<experiment_id>/ # immutable run evidence, statuses and errors
papers/jaes_v2/                 # manuscript source, generated tables and figures
docs/jaes_v2/                   # journal, novelty, extension, protocol and progress
```

Mỗi run lưu code revision hoặc diff fingerprint, config hash, environment, seeds, dataset/split/parent hashes, model/checkpoint hash, teacher ID, augmentation chain/version, selection criterion, calibration/policy hash, timestamps và status. Với local working changes, git commit alone không đủ truy nguyên code. Resume chỉ khi config/data/code compatibility pass; không silently reuse cache cũ.

Per-sample score schema tối thiểu: run_id, sample_id, parent_id, corpus_id, known speaker/generator IDs, label, condition_id, model_id, train_seed, raw logit/score, calibrated score, fixed threshold, decision, abstain flag, stage used, timings và error status. Không drop failed samples khỏi denominator mà không báo.

Tách simulation/toy/smoke khỏi official run. Trạng thái gồm `PROPOSED`, `IMPLEMENTED`, `SMOKE_TESTED`, `RUNNING`, `COMPLETED`, `FAILED`, `MISSING_DATA`, `AWAITING_COMPUTE`, `NOT_MEASURED`; định nghĩa rõ. Không đánh dấu COMPLETED khi thiếu bất kỳ model/seed/condition bắt buộc. Kết quả không tồn tại phải là missing kèm lý do, không điền 0 hoặc số dự kiến.

### 11. Kiểm thử cần thiết

Viết và chạy tests kiểm soát failure thật sự:

- Parent/variant/speaker/generator leakage theo protocol; label-conflict quarantine; metadata UNKNOWN không vượt gate; official partitions không bị ghi đè.
- Calibration/threshold/policy code không nhận test labels. Teacher cache/feature cache không vượt split. Chọn checkpoint chỉ trên development.
- Transform determinism và thứ tự; target/measured SNR; silence handling; clipping; codec delay/length/alignment; missing codec; lỗi/timeout không làm mất mẫu âm thầm.
- KD temperature và KL direction/scale đúng; teacher frozen; w bounded; paired identity đúng; group objective không collapse. Dùng analytical/toy cases có kỳ vọng độc lập với implementation.
- REAL/FAKE semantics qua TensorFlow/PyTorch/ONNX và score orientation; save/load và export numerical fidelity.
- Abstention metrics denominator đúng, matched coverage comparisons và all-abstain/zero-coverage edge cases; calibration fitted đúng partition.
- End-to-end tiny run từ manifest đến score/report; rerun aggregation không train lại; interrupted run resume không publish partial model.
- Existing baseline regression và CLI/Streamlit behavior còn chạy. Chỉ tích hợp UI sau research contract ổn; UI không phải novelty.

Smoke data chỉ kiểm tra correctness. Không biến smoke accuracy thành scientific evidence. Báo commands đã chạy, pass/fail, dependencies chưa có và những gì chưa test được.

### 12. Paper và gói bàn giao

Viết manuscript tiếng Anh mới theo template JAES chính thức hiện hành. Không sửa bản đã accept. Title mới phản ánh kết quả thật. Tạm dùng “Efficient and Reliable Audio Deepfake Detection Under Unseen Acoustic Channels”; điều chỉnh sau evidence. Theo dõi các thay đổi hướng dẫn theo ngày nộp.

Nội dung phải giải thích audio-domain mechanism, signal chain assumptions, prior art gần nhất, C1/C2/C3 có điều kiện, protocol, fairness of comparison, seed variability, ablations, failure cases và limitations. Không làm bài thành mô tả dashboard/software. Không viết results hoặc abstract có số tương lai trước khi chạy; dùng explicit placeholders trong DRAFT và bộ kiểm tra chặn submission-ready khi còn thiếu evidence.

Tạo bảng/figure từ per-sample artifacts bằng script độc lập, không hard-code kết quả trong LaTeX. Kiểm tra numerical units/percent vs fraction, sample counts, seed counts, score orientation và cohort consistency. Mỗi claim phải có dòng trong `CLAIM_EVIDENCE_MAP.csv`: claim, contribution, source run/file/hash, cohort, uncertainty, figure/table và strength supported.

Các bảng chính: extension map so với bài accept; nearest-work comparison có tiêu chí trung lập; dataset/provenance/splits; controlled clean/cross-domain results; channel chain interactions; method ablations; risk/coverage/calibration; measured deployment Pareto. Không bắt buộc mọi bảng vào main text nếu journal length không phù hợp, dùng supplement khi được phép.

Đầu ra tối thiểu:

1. JOURNAL_CHECK.md, BASELINE_AUDIT.md, EXTENSION_MATRIX.md, LITERATURE_MATRIX.csv, NOVELTY_GATE.md.
2. DATA_PROTOCOL.md, EXPERIMENT_PLAN.md, RESOURCE_PLAN.md, config schemas và validated configs.
3. Source code hoạt động, tests phù hợp, reproducible run/resume/rebuild commands và môi trường pin.
4. Artifacts và reports của những run đã thực hiện; bảng pending riêng cho việc cần data/GPU/device.
5. Manuscript DRAFT hoặc evidence-complete manuscript ở papers/jaes_v2, figures/tables và claim-evidence map.
6. Cover letter DRAFT khai báo đúng accepted prior paper; information gaps hiện rõ; không gửi.
7. RESEARCH_STATUS.md cập nhật đủ để phiên AI khác tiếp tục mà không đoán, gồm quyết định, lý do, runs đang làm, blockers và exact next commands.

### 13. Cách bắt đầu và tiêu chuẩn kết thúc

Trong lượt đầu: đọc instructions và đúng `papers/PaperID_804_final.docx`, ghi baseline hash và ba đóng góp đã có; audit repository/environment/data/artifacts; kiểm chứng literature/journal; tạo extension/novelty/experiment plan; rồi triển khai vertical slice C1 nhỏ từ manifest tới paired scores/report. Không dừng sau một plan nếu vẫn có thể code và verify. Hoàn thành C1 slice rồi mới nối objective C2 và policy C3, tránh tạo hàng loạt empty modules.

Các công việc source/audit/test nhỏ được tiến hành trong phạm vi cho phép. Chỉ hỏi khi thiếu thông tin quyết định eligibility hoặc một lựa chọn khoa học không thể tự suy ra; tiếp tục phần độc lập trong lúc chờ. Không hỏi lại từng sửa file. Không chạy full training nặng khi chưa có resource preflight; không dùng sự thiếu compute để hạ thấp tiêu chuẩn evidence.

Chỉ kết luận đạt mục tiêu ba đóng góp khi cả ba có nearest-prior-art distinction xác minh được, thực nghiệm đối chứng đúng, hiệu quả hoặc phát hiện đáng kể có uncertainty, và reproducible evidence. Negative results được báo trung thực. Nếu một contribution thất bại, phân biệt implementation bug, data confound và hypothesis failure, rồi chọn điều tra tiếp hoặc thu hẹp claim; không bẻ metric và test split để đạt mục tiêu.

Trong báo cáo cuối, tách rõ: đã triển khai; đã kiểm thử; đã đo bằng dữ liệu thật; còn giả thuyết; cần tác giả cung cấp/thiết bị/compute. Nêu ba contribution nào được hỗ trợ đến mức nào. Không tuyên bố “journal-ready”, “SOTA” hay “ba đóng góp tầm thế giới” chỉ vì code chạy hoặc paper dài hơn.

## HẾT PROMPT
