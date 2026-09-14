# original — tài liệu gốc của bài báo

Chỉ chứa tài liệu **gốc** từ tác giả / nguồn công khai, không chứa sản phẩm do nhóm viết.

- `paper/`        — PDF bài báo gốc + bản `.txt` đã extract (PyMuPDF) để đọc nhanh.
- `source-code/`  — mã nguồn gốc của tác giả (nếu có).
- `data/`         — dataset dùng trong bài báo.

## Bài báo

**HFR-MKGC: Hierarchical Fusion Reasoning with MLLMs for Multi-modal Knowledge Graph Completion**
Di Wang, Junping Du, Zhe Xue, Meiyu Liang, Guanhua Ye, Yingxia Shao (BUPT), Haisheng Li (BTBU).
*Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 40, no. 18, pp. 15815–15823, 2026.

## Nguồn

- Paper (AAAI OJS): https://ojs.aaai.org/index.php/AAAI/article/view/38613
- PDF: https://ojs.aaai.org/index.php/AAAI/article/download/38613/42575
- Video/slide trình bày của tác giả (Underline, chỉ tham khảo): https://underline.io/lecture/142842-hfr-mkgc-hierarchical-fusion-reasoning-with-mllms-for-multi-modal-knowledge-graph-completion
- Source code: **tác giả KHÔNG công khai** (bài báo không ghi link; không tìm thấy repo trên GitHub tính đến 2026-09-04). Thư mục `source-code/` để trống.
- Data: bài dùng ba benchmark chuẩn MMKGC — **DB15K**, **MKG-W**, **MKG-Y** — copy từ repo MDBGF (`../../MDBGF/original/source-code/MDBGF/data/`) sang `data/`, cùng bộ dữ liệu.
  - Nguồn gốc dataset: MMKG (Liu et al., 2019) cho DB15K; MKG-W / MKG-Y từ Xu et al., 2022 (phân phối lại qua repo MyGO/zjukg).

## Trích dẫn IEEE

[1] D. Wang, J. Du, Z. Xue, M. Liang, G. Ye, Y. Shao, and H. Li, "HFR-MKGC: Hierarchical Fusion Reasoning with MLLMs for Multi-modal Knowledge Graph Completion," in *Proc. AAAI Conf. Artif. Intell.*, vol. 40, no. 18, 2026, pp. 15815–15823.
