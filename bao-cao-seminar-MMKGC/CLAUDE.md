# Quy tắc làm việc — bao-cao-seminar-MMKGC

Folder này chứa hai bài báo về Multi-modal Knowledge Graph Completion (MMKGC) cho môn học.
Yêu cầu đầu ra của hai bài **khác nhau**, phải tuân thủ đúng.

## 1. HFR-MKGC/ — SEMINAR (trình bày trước lớp)

Bài báo: *HFR-MKGC: Hierarchical Fusion Reasoning with MLLMs for Multi-modal Knowledge Graph Completion*

Sản phẩm cần có:
- **Slide thuyết trình** (ưu tiên, đây là sản phẩm chính).
- **Báo cáo viết** kèm theo.
- Có thể thêm script/ghi chú nói cho từng slide nếu được yêu cầu.

**Bắt buộc đọc `HFR-MKGC/RULES-SEMINAR.md`** (yêu cầu chi tiết từ giảng viên: trích dẫn IEEE, ví dụ input-output, nguồn hình ảnh, tối đa 15 phút) trước khi soạn slide.

Lưu ý khi làm:
- Slide phải dễ trình bày miệng: ít chữ, nhiều hình/sơ đồ kiến trúc, mỗi slide một ý.
- Nhấn mạnh động lực (motivation), kiến trúc Hierarchical Fusion, vai trò của MLLM, kết quả thực nghiệm và hạn chế.
- Chuẩn bị phần Q&A dự kiến.

## 2. MDBGF/ — CHỈ NỘP BÁO CÁO (không thuyết trình)

Bài báo: *Not All Modalities at Once: Dynamic Dropout and Bidirectional Fusion for Robust Multi-modal Knowledge Graph Completion (MDBGF)*

Sản phẩm cần có:
- **Chỉ báo cáo viết** để nộp. Không cần slide.

Lưu ý khi làm:
- Viết đầy đủ, có cấu trúc: giới thiệu, bài toán, phương pháp (Dynamic Dropout, Bidirectional Fusion), thực nghiệm, nhận xét/đánh giá, kết luận.
- Trích dẫn đúng bài báo gốc.

## Nhóm seminar (2 thành viên)

- 25C11050 Nguyễn Đình Lộc
- 25C15003 Ngô Trương Minh Đạt

Trần Đắc Khoa **không** thuộc nhóm seminar (chỉ ở nhóm project cuối kì, repo riêng `HatDuaa/multimodal-graph-retrieval-xai`). Không ghi tên Khoa vào slide hay báo cáo seminar.

## Quy tắc chung

- Ngôn ngữ: **tiếng Việt**, giữ nguyên thuật ngữ chuyên ngành tiếng Anh khi cần (kèm giải thích lần đầu).
- Đặt file đầu ra ngay trong folder bài báo tương ứng, không lẫn giữa hai bài.
- Mỗi folder bài báo có thư mục `original/` chứa tài liệu gốc, **không sửa** nội dung bên trong:
  - `original/paper/` — PDF bài báo gốc.
  - `original/source-code/` — mã nguồn gốc của tác giả.
  - `original/data/` — dataset hoặc ghi chú link tải.
  - `original/README.md` — ghi link nguồn (arXiv, GitHub, dataset).
- Sản phẩm của nhóm (báo cáo, slide, code thử nghiệm) để **ngoài** `original/`, ngay trong folder bài báo.
- Đọc bài báo gốc trong `original/paper/` trước khi viết.
- Không bịa số liệu, không bịa kết quả thực nghiệm; nếu không tìm thấy trong bài báo thì nói rõ.
