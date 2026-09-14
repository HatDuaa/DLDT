# Project cuối kì — Học máy với dữ liệu đồ thị

## Đề đã chọn

**Project 3 — Truy vấn đa phương thức có giải thích bằng đồ thị**
(Multimodal retrieval + graph re-ranking + explainable AI). Đề gốc: [assignments/Project03_Multimodal_XAI.pdf](assignments/Project03_Multimodal_XAI.pdf).

Hai đề còn lại để tham khảo, không làm:
- [assignments/Project01_Graph_Editing.pdf](assignments/Project01_Graph_Editing.pdf) — scene graph cho chỉnh sửa ảnh. Loại vì phụ thuộc nhiều mô hình vision nặng (RelTR, InstructPix2Pix), gán nhãn tay lớn.
- [assignments/Project02_Knowledge_Tracing.pdf](assignments/Project02_Knowledge_Tracing.pdf) — knowledge tracing. An toàn nhưng không tận dụng được kiến thức MMKG từ seminar.

Lý do chọn Project 3: cấp độ 2 yêu cầu tái lập một phương pháp multimodal KG từ 2024 (NativE, AdaMF-MAT, Mixed-Curvature MMKGC), trùng với các baseline trong hai bài seminar HFR-MKGC và MDBGF; dataset MKG-W / MKG-Y / DB15K đã có sẵn trong `../bao-cao-seminar-MMKGC/HFR-MKGC/original/data/`.

## Nhóm (3 thành viên)

| MSHV | Họ tên | Vai trò ở project trước (tham khảo) |
|---|---|---|
| 25C11050 | Nguyễn Đình Lộc | Dữ liệu và tiền xử lý |
| 25C15003 | Ngô Trương Minh Đạt | Gold gán tay, baseline, cải tiến, phân tích |
| 25C15015 | Trần Đắc Khoa | Tái lập phương pháp mới (CroCoAlign) |

Phân công **đề xuất** cho Project 3 (giữ đúng thế mạnh cũ, chốt lại khi họp nhóm):

| Thành viên | Mảng chính | Sản phẩm phụ trách |
|---|---|---|
| Lộc | Dữ liệu, đồ thị, tiền xử lý | Tải FB15k-237-IMG / MKG-W, trích đặc trưng CLIP, xây đồ thị, tập chia train/val/test, kiểm soát rò rỉ (Mục 6 đề bài). Cấp độ 3: MMKG tiếng Việt tự xây + data card. |
| Đạt | Baseline, đánh giá, giải thích, phân tích | Baseline CLIP + FAISS, re-ranking α·cosine + (1−α)·graph, ablation bỏ/xáo cạnh, bộ 20 truy vấn (trực tiếp / gián tiếp), fidelity–validity–sparsity, phân tích 5 thành công / 5 thất bại. |
| Khoa | Tái lập phương pháp mới (cấp độ 2) | Chạy lại NativE hoặc AdaMF-MAT trên cùng pipeline và cách chia của cấp độ 1, so sánh với CLIP + GCN. |

Bảng phân công cuối cùng phải ghi vào báo cáo (đề yêu cầu) kèm lịch sử commit.

## Các cấp độ và điểm trần (theo đề)

| Cấp | Nội dung tóm tắt | Điểm trần |
|---|---|---|
| 1 | ≥1 tập công khai. Baseline CLIP đóng băng; đồ thị từ dữ liệu có sẵn; GCN/GAT/GraphSAGE; re-rank CLIP + Graph; ablation bỏ/xáo cạnh; hiển thị đường đi giải thích; 5 thành công / 5 thất bại. | 7.0 |
| 2 | ≥2 tập công khai. Tái lập 1 phương pháp ≥2024 (multimodal KG / graph-enhanced retrieval / explainable graph retrieval). Bộ ≥20 truy vấn có nhóm gián tiếp, báo cáo tách riêng. | 8.5 |
| 3 | Fidelity, validity, sparsity định lượng trên ≥10 truy vấn. Tự xây MMKG nhỏ tiếng Việt (50–150 thực thể, 300–1.000 triple, 200–500 ảnh), chạy phương pháp tốt nhất, phân tích chuyển giao. | 10 |
| 4 | Cải tiến riêng (graph-aware contrastive loss, re-ranking thích nghi, chọn đường đi giải thích…), video, hoặc khảo sát người dùng. Cần giả thuyết + ablation + ≥2 tập. | +1.0 – 2.0 |

Metric bắt buộc: Recall@1/5/10, MRR; so CLIP thuần với CLIP + Graph trên cùng cách chia; ≥3 seed, báo cáo mean ± std.

## Quyết định kỹ thuật ban đầu

- **Nhánh A (text-to-image)** là bắt buộc. Image-to-text làm sau nếu còn thời gian.
- **Tập cấp độ 1: FB15k-237-IMG** (KG chuẩn có ảnh gắn thực thể, kho MKGformer). Chọn thay Visual Genome vì đồ thị có sẵn dạng triple, gần dữ liệu MKG-W của cấp độ 2, pipeline liền mạch.
- **Tập cấp độ 2: MKG-W** (đã có) + phương pháp tái lập ưu tiên **NativE (SIGIR 2024)** hoặc **AdaMF-MAT (LREC-COLING 2024)**, chọn theo cái nào chạy lại được trước.
- **Điểm đồ thị phải mang giải thích ngay từ đầu**: thiết kế theo đường đi truy vấn → khái niệm → ảnh hoặc attention trên cạnh (GAT), để cấp độ 3 đo fidelity bằng cách bỏ đúng cạnh đã dùng. Không dùng cách vẽ đường đi hậu kiểm (đề cấm).
- Đặc trưng CLIP trích một lần, lưu `.npy`; FAISS cho top-k; PyTorch Geometric cho GNN.

## Sản phẩm phải nộp

- Mã nguồn (tiền xử lý, trích đặc trưng, tạo đồ thị, huấn luyện, đánh giá) + `requirements.txt` + hướng dẫn chạy lại.
- Tệp chia dữ liệu hoặc seed + mã tạo split, mô tả chống rò rỉ.
- Báo cáo kỹ thuật 12–18 trang, slide, bảng phân công.
- Notebook hoặc giao diện nhỏ: nhập truy vấn → top-k → hiển thị đường đi giải thích.
- Phân tích ≥5 truy vấn thành công, ≥5 thất bại.
- Cấp độ 2: bộ ≥20 truy vấn + cách xây. Cấp độ 3: MMKG tự xây + data card.
- Demo chạy trực tiếp khi vấn đáp, video dự phòng ≤5 phút. Trình bày ≤20 phút.

## Cấu trúc thư mục dự kiến

```
final-project/
├── README.md
├── CLAUDE.md                # quy tắc làm việc trong folder này
├── assignments/                  # 3 đề gốc (PDF), không sửa
├── data/                    # dữ liệu tải về + tập chia (không commit dữ liệu lớn)
├── src/                     # mã nguồn
├── notebooks/               # demo, phân tích
├── experiments/             # log, kết quả theo seed
├── report/                  # báo cáo LaTeX, slide (nội dung tiếng Việt)
└── vietnamese-mmkg/         # MMKG tiếng Việt tự xây (cấp độ 3) + data card
```

## Trạng thái

- 2026-09-09: chốt đề Project 3, chưa bắt đầu code. Việc tiếp theo: kế hoạch chi tiết theo cấp độ, tải FB15k-237-IMG, dựng baseline CLIP.
