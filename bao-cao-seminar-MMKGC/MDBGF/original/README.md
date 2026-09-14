# original — tài liệu gốc của bài báo

Chỉ chứa tài liệu **gốc** từ tác giả / nguồn công khai, không chứa sản phẩm do nhóm viết.

- `paper/`        — PDF bài báo gốc + bản `.txt` đã extract (PyMuPDF) để đọc nhanh.
- `source-code/`  — mã nguồn gốc của tác giả (clone từ GitHub, giữ nguyên, không sửa).
- `data/`         — dataset dùng trong bài báo.

## Bài báo

**Not All Modalities at Once: Dynamic Dropout and Bidirectional Fusion for Robust Multi-modal Knowledge Graph Completion**
Jiashun Peng, Fu Zhang, Hongzhi Chen, Jingwei Cheng, Yingsong Ning, Xiaoke Wang — Northeastern University (Shenyang).
*Findings of the Association for Computational Linguistics: ACL 2026*, pp. 17928–17940, July 2026.

## Nguồn

- Paper (ACL Anthology): https://aclanthology.org/2026.findings-acl.890/
- PDF: https://aclanthology.org/2026.findings-acl.890.pdf
- Source code (tác giả): https://github.com/ferryman-ship/MDBGF
  - Clone shallow ngày 2026-09-04 vào `source-code/MDBGF/`.
  - Code xây dựng trên nền MyGO (zjukg). Repo kèm sẵn `data/` (triples) và `tokens/` (token BEiT cho ảnh, BERT/RoBERTa/LLaMA cho text, một số file zip cần giải nén trước khi train).
- Data: ba benchmark chuẩn MMKGC — **DB15K**, **MKG-W**, **MKG-Y** — copy từ `source-code/MDBGF/data/` sang `data/`.
  - Nguồn gốc dataset: MMKG (Liu et al., 2019) cho DB15K; MKG-W / MKG-Y từ Xu et al., 2022 (được phân phối lại qua repo MyGO/zjukg).

## Trích dẫn IEEE

[1] J. Peng, F. Zhang, H. Chen, J. Cheng, Y. Ning, and X. Wang, "Not All Modalities at Once: Dynamic Dropout and Bidirectional Fusion for Robust Multi-modal Knowledge Graph Completion," in *Findings of the Association for Computational Linguistics: ACL 2026*, 2026, pp. 17928–17940.
