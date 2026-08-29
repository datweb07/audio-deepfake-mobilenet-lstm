# Audio Deepfake Detection - MobileNetV3Small + LSTM

Hệ thống phân loại nhị phân audio với quy ước thống nhất:

- `REAL = 0`
- `FAKE = 1`
- đầu ra sigmoid là `P(FAKE)`
- `P(FAKE) >= threshold` được phân loại là `FAKE`

Pipeline chính là: audio 3 giây -> 6 segment theo đúng thứ tự thời gian -> Mel spectrogram RGB 224x224 -> `TimeDistributed(MobileNetV3Small)` -> LSTM -> sigmoid.

## 1. Setup

Khuyến nghị dùng Python 3.11 và một virtual environment mới. Không cài requirements của ba repo reference.

```powershell
cd D:\audio-deepfake-mobilenet-lstm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra môi trường:

```powershell
python -m pip check
python -c "import tensorflow, librosa, cv2, sklearn, streamlit; print(tensorflow.__version__)"
```

TensorFlow tự dùng GPU tương thích nếu phát hiện được. Không có GPU thì project vẫn chạy bằng CPU, nhưng training chậm hơn đáng kể.

## 2. Dataset

Chỉ hai thư mục sau được scan làm dataset production:

```text
data\REAL\
data\FAKE\
```

Định dạng hỗ trợ: `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`. Mỗi file được resample về 22,050 Hz, mono và lấy tối đa 3 giây đầu. File ngắn hơn được zero-pad.

Ba thư mục sau chỉ là nguồn tham khảo, không được scan làm dataset và không được import vào runtime:

```text
deepfake-audio-detection\
enhancing-deepfake-detection-using-mobilenet-lstm-hybrid-model-main\
mobilenetv3.pytorch\
```

Split được tạo lại một cách deterministic với seed 42 theo tỷ lệ 70% train, 15% validation, 15% test. Original files được split trước; augmentation chỉ chạy trên train.

## 3. Train

```powershell
python train.py
```

Training gồm:

1. Một warm-up stage nội bộ: đóng băng MobileNetV3Small và train LSTM/classifier với learning rate `1e-4`.
2. Pipeline tự động khôi phục best validation state, mở phần cuối backbone, giữ BatchNormalization frozen, compile lại và fine-tune với learning rate `1e-5`.
3. Một checkpoint `val_loss` dùng xuyên suốt toàn lifecycle chọn global-best weights; checkpoint chỉ là artifact tạm trong `outputs/checkpoints/`.
4. Global-best model được save/load-verify rồi xuất thành đúng một production model.
5. Validation probabilities của final model được dùng để calibrate threshold theo F1. Test set không tham gia training, early stopping, checkpoint selection hay calibration.

Public artifacts:

```text
models\lava_mobilenetv3_lstm.keras
models\best_threshold.txt
models\model_metadata.json
outputs\plots\training_history.png
```

Warm-up và fine-tuning là hai stage của **một training run**, không phải hai detector. Sau khi production model mới save/load thành công, các model stage cũ được chuyển sang `outputs\legacy_models\`; runtime không có fallback sang các artifact này.

Nếu production model chưa tồn tại, evaluate, predict và UI đều báo:

```text
Production model not found. Run: python train.py
```

## 4. Evaluate

Chỉ chạy sau khi đã train lại:

```powershell
python evaluate.py
```

Script dùng test split độc lập và threshold đã tune trên validation. Output gồm Accuracy, Precision, Recall, F1, Macro F1, ROC-AUC từ raw `P(FAKE)`, EER, confusion matrix và classification report.

## 5. Predict

```powershell
python predict.py --audio "data\REAL\biden-original.wav"
python predict.py --audio "data\FAKE\biden-to-linus.wav"
```

Output gồm prediction, confidence của class được chọn, raw `P(FAKE)` và threshold thực tế.

## 6. Streamlit

```powershell
python -m streamlit run app.py
```

Mở `http://localhost:8501`. UI hỗ trợ upload/playback, waveform, Mel spectrogram, REAL/FAKE, confidence, raw `P(FAKE)` và threshold. UI không hiển thị class prediction giả cho từng segment vì classifier chỉ dự đoán toàn sequence.

## 7. Retraining với dataset mới

1. Thay nội dung `data\REAL` và `data\FAKE`.
2. Chạy lại `python train.py`.
3. Final production model, metadata, một lifecycle plot và threshold sẽ được cập nhật.
4. Split được regenerate deterministic từ danh sách file mới.
5. Chạy `python evaluate.py`, sau đó smoke-test `predict.py` và Streamlit.

Không dùng threshold cũ với model mới và không dùng model cũ với preprocessing mới.

## 8. Production artifact contract cho LAVA

Mỗi detector chỉ được xuất một final artifact có thể so sánh/deploy. Với detector hiện tại, contract là:

```text
MobileNetV3Small-LSTM -> models\lava_mobilenetv3_lstm.keras
```

Các architecture tương lai cũng phải dùng cùng nguyên tắc “one detector -> one final artifact”. Warm-up, fine-tuning, pruning hoặc calibration là chi tiết nội bộ của training algorithm, không tạo thêm production-model choices cho UI hay benchmark runner.
