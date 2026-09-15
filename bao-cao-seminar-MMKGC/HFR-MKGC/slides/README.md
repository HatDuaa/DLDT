# Slide seminar HFR-MKGC

- `seminar-slides.html` — slide chính, tự chứa, mở bằng trình duyệt. Phím ← → chuyển trang, **N** bật ghi chú nói, **P** in ra PDF.
- `25C11050_25C15003_HFR-MKGC-slides.pdf` — bản in sẵn (20 trang, 16:9) để nộp hoặc dự phòng.
- `study-guide.html` — bản học: từng slide kèm phần nói bên cạnh, cuối có Q&A. Mở bằng trình duyệt.
- `speaker-script.md` — script nói dạng văn bản (nguồn của study-guide). Sửa xong chạy `python build_study_guide.py`.
- `seminar-slides-template.html` + `build_slides.py` — nguồn. Sửa template hoặc ghi chú nói trong `NOTES` rồi chạy:

```bash
python build_slides.py
```

Hình gốc lấy từ `../paper-vi/figures/` (cắt từ PDF bài báo). Hình do AI tạo (Claude) ghi nguồn là link artifact.
