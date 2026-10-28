# Báo cáo môn học: HFR-MKGC và MDBGF

Một báo cáo chung cho cả hai bài báo theo yêu cầu trong [`../RULES-REPORT.md`](../RULES-REPORT.md): template IEEE conference, tối đa 25 trang, một file PDF duy nhất.

## File

- `mmkgc-report.tex` — nguồn LaTeX (XeLaTeX, IEEEtran, tiếng Việt).
- `references.bib` — tài liệu tham khảo, giữ tiếng Anh theo bản gốc.
- `figures/` — hình lấy từ bài báo gốc (cắt từ PDF). Hình tự vẽ nằm trực tiếp trong `.tex` bằng TikZ/pgfplots.
- `build.sh` — biên dịch và tạo file nộp.
- `25C11050-25C15003.pdf` — file nộp (tên theo MSHV, bản build mới nhất).

## Biên dịch

```bash
bash build.sh
```

Cần MiKTeX hoặc TeX Live có XeLaTeX, font Times New Roman, các gói IEEEtran, polyglossia, tikz, pgfplots, booktabs.

## Trạng thái

| Phần | Trạng thái |
|---|---|
| I. Giới thiệu chung (bài toán, đánh giá, dữ liệu) | xong |
| II. Phần A: HFR-MKGC | xong bản đầu |
| III. Phần B: MDBGF | xong bản đầu (chưa có tái lập, chừa chỗ link WandB/GitHub) |
| IV. Đối chiếu và kết luận | mới có đoạn sơ bộ, cần viết hoàn chỉnh |

Chỗ còn trống được đánh dấu màu đỏ trong PDF bằng lệnh `\todo{}`. Trước khi nộp phải xoá hết các chỗ đó.

## Lưu ý khi viết tiếp

- Viết bằng lời của nhóm, không chép câu chữ bài gốc. Bản dịch trong `../HFR-MKGC/paper-vi/` chỉ dùng để đọc, không bê vào báo cáo.
- Số liệu lấy từ bài báo thì ghi rõ bảng/hình nguồn trong caption. Số liệu nhóm tự tính phải nói rõ.
- Nếu chạy tái lập MDBGF: log bằng Weights & Biases, project công khai, gắn link ngay cạnh bảng kết quả tương ứng; link repo GitHub để ở phần thực nghiệm.
