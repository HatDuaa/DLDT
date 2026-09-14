# Slide seminar HFR-MKGC

- `seminar-slides.html` — slide chính, tự chứa, mở bằng trình duyệt. Phím ← → chuyển trang, **N** bật ghi chú nói, **P** in ra PDF.
- `seminar-slides.pdf` — bản in sẵn (22 trang, 16:9): 20 slide trình bày + 2 slide phụ lục (A: Table 1 đầy đủ, B: độ nhạy siêu tham số). Phụ lục chỉ dùng khi hỏi đáp, không tính vào 15 phút.
- `seminar-slides-template.html` + `build_slides.py` — nguồn. Sửa template hoặc ghi chú nói trong `NOTES` rồi chạy:

```bash
python3 build_slides.py
```

Script không bắt buộc Pillow: có thì ảnh được nén JPEG, không có thì nhúng PNG gốc.

## Nguồn hình

- Fig. 1, 2, 3, 4 lấy từ `../paper-vi/figures/` (cắt từ PDF bài báo), ghi nguồn [1].
- Hình tự vẽ ghi hai link trong slide tài liệu tham khảo: bản vẽ (`SRC_SELF`, artifact) và cuộc trò chuyện với AI đã hướng dẫn tạo/chỉnh sửa (`SRC_CHAT`). Cả hai đặt ở đầu `build_slides.py`.

## Xuất PDF

Cách đơn giản: mở HTML bằng Chrome, nhấn **P**, chọn "Save as PDF", khổ giấy sẽ tự là 1280×720.

## Gợi ý thời gian (15 phút)

Slide 1–13 (bài toán, kiến trúc): ~8 phút. Slide 14–19 (thực nghiệm, nhận xét, kết luận): ~7 phút. Nếu thiếu giờ, lướt nhanh slide 6 (Figure 2 gốc) vì slide 13 đã có luồng dữ liệu tự vẽ.
