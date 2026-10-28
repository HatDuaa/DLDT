# Tái lập MDBGF

Thư mục này chứa mã để chạy lại các thực nghiệm của bài *Not All Modalities at Once: Dynamic Dropout and Bidirectional Fusion for Robust Multi-modal Knowledge Graph Completion* (Findings of ACL 2026) trên bộ dữ liệu MKG-W. Kết quả và phân tích nằm trong báo cáo, không nằm ở đây.

Mã gốc của tác giả ([ferryman-ship/MDBGF](https://github.com/ferryman-ship/MDBGF), commit `c512d79`) được dùng nguyên trạng. Repo này chỉ thêm:

| File | Vai trò |
|---|---|
| `prepare_data.py` | Giải nén token ảnh, tạo `tokens/textual.pth` từ `bert-base-uncased` (repo tác giả không kèm file này). |
| `train.py` | Script huấn luyện, dựa trên `train_mygo_fgc.py` của tác giả, có ghi Weights & Biases và chạy tiếp từ checkpoint. |
| `mdbgf_colab.ipynb` | Notebook chạy toàn bộ trên Google Colab. |

## Môi trường

- Google Colab Pro, GPU **A100 40 GB** hoặc **L4 24 GB**. Theo bài báo, mô hình cần khoảng 21 GB bộ nhớ GPU, nên GPU T4 16 GB của bản Colab miễn phí không đủ.
- Python và PyTorch có sẵn trên Colab; cài thêm các gói trong `requirements.txt`.
- Tài khoản [Weights & Biases](https://wandb.ai) và API key.
- Google Drive còn trống khoảng 1 GB cho checkpoint.

## Cách chạy trên Colab

1. Mở `mdbgf_colab.ipynb` trên Colab, chọn GPU A100 hoặc L4.
2. Thêm WandB API key vào Colab Secrets với tên `WANDB_API_KEY`.
3. Chạy lần lượt các ô 1 đến 6. Ô 6 chạy thử 10 epoch để đo thời gian và bộ nhớ.
4. Ở ô 7, chọn `RUN` rồi chạy. Mỗi lượt dừng sau 11 giờ và lưu checkpoint vào Drive. Khi phiên Colab bị ngắt hoặc hết giờ, chạy lại các ô 1–5 rồi ô 7 với cùng `RUN` và `EPOCHS`; script tự chạy tiếp từ checkpoint cuối và ghi tiếp vào cùng run trên WandB.

## Chạy bằng dòng lệnh

```bash
# chạy trong thư mục bao-cao-seminar-MMKGC/MDBGF/reproduce
git clone https://github.com/ferryman-ship/MDBGF.git && git -C MDBGF checkout c512d79
pip install -r requirements.txt
python prepare_data.py --src MDBGF --data MKG-W
wandb login
python train.py --src MDBGF --out runs/main_code --md code --num_epoch 1550
```

## Các lần chạy

Mọi lần chạy dùng siêu tham số của tác giả cho MKG-W (`run.sh`): batch 2048, `lr 6e-4`, `dim 256`, 16 token ảnh, 24 token văn bản, hạt giống 2024. Riêng số epoch, nhóm chạy **1550** thay vì 5000 của bài (`--num_epoch 1550`, đã đặt sẵn trong notebook); muốn chạy đúng như bài thì bỏ tham số này.

| `RUN` | Tham số thêm | Ứng với |
|---|---|---|
| `main_code` | `--md code` | Kết quả chính, đúng như mã tác giả công khai |
| `main_paper` | `--md paper` | Kết quả chính, lịch rụng mô thức theo công thức (1) của bài |
| `no_md` | `--md none` | Bỏ rụng mô thức |
| `no_bgf` | `--no_bgf` | Bỏ hai cổng hợp nhất |
| `no_ahns` | `--mu 0` | Bỏ lấy mẫu âm lai |

## Các tham số chính của `train.py`

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--src` | (bắt buộc) | Thư mục mã nguồn của tác giả |
| `--out` | (bắt buộc) | Thư mục lưu checkpoint; nếu đã có `last.ckpt` thì chạy tiếp |
| `--num_epoch` | `5000` | Tổng số epoch; nhóm dùng `1550` |
| `--md` | `code` | Cách rụng mô thức: `code`, `paper` hoặc `none` |
| `--alpha` | `0.3` | Xác suất rụng tối đa |
| `--no_bgf` | tắt | Bỏ hai cổng hợp nhất |
| `--mu` | `0.01` | Trọng số hàm mất mát lấy mẫu âm lai; `0` là bỏ |
| `--valid_epoch` | `50` | Số epoch giữa hai lần đánh giá và lưu checkpoint |
| `--max_hours` | `0` | Dừng và lưu checkpoint sau ngần này giờ (`0` là không giới hạn) |
| `--no_wandb` | tắt | Không ghi lên WandB |

Các siêu tham số còn lại giữ tên và giá trị như script của tác giả (`python train.py -h` để xem).

## Kết quả được ghi ở đâu

- **WandB:** project `mdbgf-reproduce`, mỗi `RUN` là một run. Mỗi epoch ghi loss, thời gian và bộ nhớ GPU; mỗi 50 epoch ghi MRR, Hits@1/3/10 trên tập kiểm định và kiểm thử. Phần *summary* của run có hai mục: `best_by_valid` (epoch có MRR kiểm định cao nhất) và `best_by_test` (epoch có MRR kiểm thử cao nhất, cách chọn trong mã tác giả).
- **Drive:** `mdbgf-runs/<RUN>/last.ckpt`.
- **Phiên bản thư viện và phần cứng:** WandB tự ghi phiên bản Python, danh sách gói (`requirements.txt` của môi trường) và loại GPU vào tab *Overview* và *Files* của mỗi run.

Để giảng viên xem được, chuyển project trên WandB sang chế độ **Public** (Project → Settings → Visibility).
