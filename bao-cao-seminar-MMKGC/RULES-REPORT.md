# Yêu cầu nộp báo cáo (từ giảng viên)

Áp dụng cho **một báo cáo chung** gồm cả hai bài báo: HFR-MKGC (đã seminar) và MDBGF (chỉ báo cáo).

## Định dạng nộp

- **1 file PDF duy nhất**, tên theo MSHV nối bằng gạch nối: `25C11050-25C15003.pdf`.
- Dưới **20 MB**, tối đa **25 trang** tính cả bìa, tài liệu tham khảo và phụ lục.
- Template **IEEE conference** (A4): bản Word <https://ieee-org.widen.net/content/ge5anzdecd/original/conference-template-a4.docx>, bản LaTeX <https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn>.
- Một ngôn ngữ duy nhất (tiếng Việt hoặc tiếng Anh), không pha tạp.

## Nội dung

- Hai bài báo là **hai phần của một báo cáo**, không phải hai báo cáo ghép lại; giới hạn trang tính trên tổng.
- **Không sao chép** câu chữ của bài báo gốc hay tài liệu công khai; phải diễn đạt cách hiểu của nhóm bằng lời của mình.
- Được vẽ lại / trực quan hóa khái niệm, nhưng cân đối kích thước hình với giới hạn trang; hình đính kèm phải đủ độ phân giải.

## Thực nghiệm tái lập

- Quá trình tái lập ghi nhận bằng **Weights & Biases**; link chia sẻ phải **xem công khai**.
- Trong báo cáo, **mỗi phần kết quả gắn link WandB tương ứng** (kết quả chính → link run chính; ablation → link run ablation; ...).
- Project WandB ở chế độ Teams: dùng cross-project report, xem <https://docs.wandb.ai/models/reports/cross-project-reports>.

## Mã nguồn

- Quản lý trên **GitHub**, tên repo tùy chọn, **dẫn link trong báo cáo** ở phần tương ứng.
- `README.md` của repo phải nói rõ cách chạy lại kết quả: môi trường, phần cứng, câu lệnh, file Python/Notebook nào.
- **Không** đưa kết quả thực nghiệm hay mô tả giải pháp vào README (những phần đó thuộc báo cáo).

## Checklist trước khi nộp

- [ ] Một PDF, tên `25C11050-25C15003.pdf`, < 20 MB, ≤ 25 trang.
- [ ] Đúng template IEEE, một ngôn ngữ thống nhất.
- [ ] Có cả hai phần HFR-MKGC và MDBGF, viết liền mạch.
- [ ] Không copy câu chữ bài gốc; hình tự vẽ có nguồn.
- [ ] Mỗi bảng/kết quả tái lập có link WandB công khai.
- [ ] Link GitHub repo trong báo cáo; README hướng dẫn chạy lại, không chứa kết quả.
