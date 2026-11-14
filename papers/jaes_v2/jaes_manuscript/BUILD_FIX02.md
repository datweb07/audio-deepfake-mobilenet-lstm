# Biên dịch JOURNAL FIX02

## Overleaf

1. Chọn **New Project → Upload Project** và tải ZIP lên.
2. Vào **Settings → Compiler**, chọn **pdfLaTeX**.
3. Chọn **main_review.tex** làm Main document và bấm **Recompile**.

`main_review.tex` tạo bản A4 một cột gửi giảng viên. `main.tex` dùng cùng nội dung với template JAES hai cột và còn các trường tác giả cần điền. Rà checklist nộp bài trước khi dùng bản JAES chính thức.

## Biên dịch local

Chạy từ thư mục giải nén với TeX Live hoặc MiKTeX và latexmk đã cài.

```powershell
latexmk -pdf -halt-on-error -interaction=nonstopmode main_review.tex
```

Gói chứa LaTeX có thể sửa, các hình vector, bảng, bibliography và thư viện cần thiết. Bản review đã biên dịch bằng pdfLaTeX/BibTeX ra 33 trang. Cách dàn dòng và vị trí float có thể khác PDF FIX02 đã duyệt. Nội dung được dựng từ FIX02. Các sửa nhỏ trong lần dựng gồm nhãn REAL false alarms/FAKE misses, đơn vị 21 bản ghi cho từng speaker LA_0045 và LA_0046, và phạm vi 13.440 lượt nguồn thuộc các arm 640. Lời cảm ơn UEH giữ nguyên. Bản LaTeX dùng nội dung review. Kiểm tra yêu cầu khai báo journal trước khi nộp.
