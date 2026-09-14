# Tóm tắt bài báo HFR-MKGC

**HFR-MKGC: Hierarchical Fusion Reasoning with MLLMs for Multi-modal Knowledge Graph Completion**
D. Wang, J. Du, Z. Xue, M. Liang, G. Ye, Y. Shao, H. Li — AAAI 2026 (vol. 40, pp. 15815–15823).
Nguồn: [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/38613) · PDF gốc trong `original/paper/`.
Mọi số liệu trong file này lấy trực tiếp từ bài báo (ghi số trang PDF ở mỗi mục). Tác giả **không công khai code**.

---

## 0. Kiến thức nền cần nắm trước khi đọc

| Thuật ngữ | Giải thích ngắn |
|---|---|
| **Knowledge Graph (KG)** | Đồ thị tri thức, lưu sự thật dưới dạng bộ ba **(h, r, t)** = (head entity, relation, tail entity). Ví dụ: (Gwen Stefani, voice type, Mezzo-soprano). |
| **KGC – Knowledge Graph Completion** | Dự đoán thực thể còn thiếu trong bộ ba không đầy đủ (h, r, ?) hoặc (?, r, t). Còn gọi là *link prediction*. |
| **MMKG / MMKGC** | KG mà mỗi thực thể có thêm **mô tả văn bản** và **ảnh** (multi-modal). MMKGC = KGC trên MMKG, dùng thêm text + ảnh để đoán tốt hơn. |
| **Modality (mô thức)** | Một "loại" thông tin của thực thể. Bài này dùng 3 loại: **S** (structural – cấu trúc đồ thị), **T** (textual – văn bản), **I / V** (image/visual – ảnh). |
| **Scoring function f(h, r, t)** | Hàm cho điểm "độ hợp lý" của một bộ ba. Xếp hạng tất cả ứng viên theo điểm, chọn cao nhất. |
| **RotatE** | Một scoring function: biểu diễn embedding trong **không gian số phức**, quan hệ r là một **phép xoay** (rotation) góc θ. Bộ ba đúng khi `h ∘ r ≈ t`. |
| **Negative sampling** | Tạo bộ ba **sai** (thay h hoặc t bằng thực thể khác) để mô hình học phân biệt đúng/sai. **Hard negative** = mẫu sai nhưng "trông giống đúng", khó phân biệt, giúp mô hình học tốt hơn. |
| **LLM / MLLM** | Large Language Model / Multimodal LLM (nhận cả ảnh + text). Bài này dùng **LLaVA**. |
| **LoRA** | Low-Rank Adaptation: fine-tune LLM bằng cách chỉ học một số ma trận nhỏ (hạng thấp) thay vì toàn bộ tham số, rẻ hơn nhiều. |
| **CLIP / BERT** | Bộ mã hoá (encoder) pre-trained: CLIP cho ảnh, BERT cho văn bản → ra vector đặc trưng. |
| **Adversarial training (GAN-style)** | Hai mạng "đấu" nhau: **generator** cố tạo mẫu sai khó, **discriminator** (ở đây chính là scoring function) cố phân biệt. Cả hai cùng tốt lên. |

---

## 1. Bài báo nói về vấn đề gì, giải quyết việc gì

### Vấn đề
KG luôn **thiếu** sự thật (incomplete). KGC truyền thống (TransE, RotatE, DistMult, ComplEx…) chỉ dùng cấu trúc đồ thị, bỏ qua thông tin text/ảnh. MMKGC ra đời để tận dụng thêm các mô thức này.

Tác giả chỉ ra **2 điểm yếu** của MMKGC hiện tại (trang 1):

1. **Thiếu fusion có hướng dẫn bởi quan hệ (relation-guided fusion).** Đa số phương pháp trộn các mô thức với trọng số **cố định / bằng nhau** cho mọi quan hệ. Nhưng thực tế mức quan trọng của từng mô thức **thay đổi theo quan hệ**: quan hệ *voice type* thì ảnh gần như vô dụng, text lại rất hữu ích; quan hệ *màu lông* thì ngược lại. Fusion "mù quan hệ" (relation-agnostic) dễ bị nhiễu bởi mô thức không liên quan.
2. **LLM-based KGC kém linh hoạt.** Các phương pháp dùng LLM cho KGC mới chỉ khai thác text + cấu trúc, fusion tĩnh, **chưa tận dụng ảnh** và không thích nghi theo từng bộ ba.

### Giải pháp đề xuất: HFR-MKGC
Framework gồm **3 module** (Figure 2, trang 3):

1. **Relation-Guided Hierarchical Modal Fusion (RHF)** – fusion phân cấp 2 tầng: tầng 1 trộn *bên trong* mô thức ảnh (ảnh thô + caption do MLLM sinh), tầng 2 trộn *giữa* 3 mô thức S/I/T với trọng số **tính từ quan hệ r**.
2. **MLLMs-Enhanced Knowledge Reasoning and Scoring (MER)** – fine-tune LLaVA bằng LoRA để nó **đoán thẳng thực thể thiếu** từ prompt (text + láng giềng + ảnh + quan hệ); kết quả được **trộn bằng cổng (gate)** vào embedding rồi chấm điểm bằng RotatE.
3. **Multi-Modal Negative Sample Optimization (MNO)** – tạo **hard negative** đa mô thức: MLLM viết lại mô tả text, xoay + thêm nhiễu vào đặc trưng ảnh; huấn luyện **đối kháng** (adversarial) với gradient penalty.

### Đóng góp tác giả tự nêu (trang 2)
- Framework mới với intra-visual fusion + relation-guided cross-modal weighting, tăng robust qua adversarial training.
- Chiến lược MER: MLLM fine-tuned suy luận theo từng bộ ba, kết quả tích hợp vào scoring.
- SOTA trên 3 benchmark so với **14 baseline**.

---

## 2. Các nghiên cứu liên quan (Related Work, trang 2)

### 2.1 MMKGC – 3 thế hệ phương pháp fusion

| Nhóm | Đại diện | Ý tưởng |
|---|---|---|
| **Early fusion đơn giản** | Mousselly-Sergieh et al. 2018; IKRL (Xie 2017); TransAE (Wang 2019) | Nối / cộng trung bình embedding các mô thức, hoặc autoencoder chung. |
| **Attention / Gating** | **MKGformer** (Chen 2022, SIGIR): hybrid transformer, fusion phân cấp lọc nhiễu ảnh. **RSME** (Wang 2021): gate tắt mô thức vô dụng. **MCKGC** (Gao 2025, AAAI): nhúng vào nhiều không gian hình học (mixed-curvature) + gating phân cấp. | Học trọng số trộn, nhưng vẫn **không phụ thuộc quan hệ**. |
| **Negative sample / adversarial** | **MACO** (Zhang 2023): thêm nhiễu mô phỏng thiếu mô thức. **AdaMF-MAT** (Zhang 2024c, LREC-COLING): trọng số fusion thích nghi học bằng đối kháng. **NativeE** (Zhang 2024a, SIGIR): mở rộng sang số, audio, video. **SNAG** (Chen 2025a): noise-powered. **DHNS** (Chen 2025b): diffusion sinh negative. **APKGC** (Jian 2025, AAAI): noise + attention penalty. **MyGO** (Zhang 2025, AAAI): token hoá mô thức fine-grained. | Sinh mẫu âm khó để tăng tổng quát hoá. Đây là nhóm **baseline mạnh nhất** của bài (NativeE là best baseline). |

→ **Khoảng trống**: hầu hết relation-agnostic → HFR-MKGC đưa quan hệ vào trọng số fusion.

### 2.2 LLM cho suy luận tri thức trên KG
- **GAugLLM** (Fang 2024): prompt augmentation cho graph có text.
- Dịch cấu trúc KG sang text cho LLM: entity alignment (Jiang 2024; AutoAlign – Zhang 2023), KGC (**KoPA** – Zhang 2024b "Making LLMs perform better in KGC"; Yao 2025).
- **LP-DIXIT** (Barile 2025): dùng LLM đánh giá giải thích link prediction.
- **LoLLM** (Pan 2025): LLM + structure embedding cho KG thưa.
- **HOLMES** (Panda 2024): KG chưng cất đưa vào LLM cho QA.
- MMKG + LLM: **UniMEL** (Liu 2024) entity linking; **MuKDC** (Li 2024) few-shot KGC; **CATS** (Li 2025) inductive KGC.

→ **Khoảng trống**: LLM-based mới dùng text + cấu trúc, fusion tĩnh → HFR-MKGC dùng **MLLM** (có ảnh) và **gate động** theo từng bộ ba.

> Gợi ý cho slide: vẽ 3 paradigm như Figure 1 của bài: (a) MMKGC truyền thống: encoder → fusion → score; (b) LLM-based KGC: KG → prompt → LLM → đáp án; (c) HFR-MKGC: kết hợp cả hai + hard negative.

---

## 3. Dữ liệu (Table 2, trang 6)

Ba benchmark MMKGC chuẩn, mỗi thực thể có **mô tả văn bản** và **ảnh** đi kèm. Dùng chia train/valid/test chuẩn của các công trình trước.

| Dataset | Nguồn gốc | #Entity | #Relation | #Train | #Valid | #Test | #Image | #Text |
|---|---|---|---|---|---|---|---|---|
| **MKG-W** | Xu et al. 2022 (từ Wikidata) | 15 000 | 169 | 34 196 | 4 276 | 4 274 | 14 463 | 14 123 |
| **MKG-Y** | Xu et al. 2022 (từ YAGO) | 15 000 | 28 | 21 310 | 2 665 | 2 663 | 14 244 | 12 305 |
| **DB15K** | Liu et al. 2019 (từ DBpedia) | 12 842 | 279 | 79 222 | 9 902 | 9 904 | 12 818 | 9 078 |

**Nhận xét về phân bổ** (tự suy từ bảng, không phải tác giả viết):
- Tỉ lệ chia ≈ **80 / 10 / 10** ở cả ba tập.
- **Không phải thực thể nào cũng có đủ ảnh/text**: MKG-W ~96% có ảnh, ~94% có text; MKG-Y ~95% ảnh, chỉ ~82% text; DB15K ~99.8% ảnh nhưng chỉ ~71% text → DB15K "thiếu text" nhiều nhất. Đây là lý do fusion phải chịu được mô thức thiếu/nhiễu.
- **MKG-Y** rất ít quan hệ (28) nhưng số thực thể bằng MKG-W → mỗi quan hệ dày, dễ hơn về cấu trúc (điểm RotatE thuần đã ~36.7 MRR).
- **DB15K** nhiều quan hệ nhất (279) và nhiều bộ ba nhất (~99k) → đồ thị dày nhất.

**Định dạng file thực tế** (xem `original/data/`, copy từ repo MDBGF, cùng benchmark):
- `train.txt / valid.txt / test.txt`: mỗi dòng **3 cột tab** `head_id  relation_id  tail_id` (đã mã hoá số).
- `entities.txt`, `relations.txt`: danh sách id.
- `entity2id.txt` (MKG-W, MKG-Y): ánh xạ **tên thực thể → id**, ví dụ `http://www.wikidata.org/entity/Q370252 0` (MKG-W dùng Wikidata QID) hoặc `Gilles_Mbang_Ondo 0` (MKG-Y dùng tên YAGO).
- Ảnh và text gốc **không nằm trong folder này**; các repo MMKGC thường phát hành sẵn **đặc trưng đã encode** (vector CLIP/BEiT cho ảnh, BERT cho text) thay vì file ảnh thô.

**Input – Output của bài toán** (dùng cho slide theo yêu cầu giảng viên):
- Input: bộ ba thiếu `(Gwen Stefani, voice type, ?)` + ảnh Gwen Stefani + mô tả text "American singer, songwriter…" + các láng giềng trong KG.
- Output: bảng xếp hạng toàn bộ thực thể ứng viên, top-1 = `Mezzo-soprano` (case study Figure 4, trang 7).

---

## 4. Mô hình nền (base) có không? Có, gồm 3 phần

| Thành phần | Mô hình nền được dùng | Vai trò trong HFR-MKGC |
|---|---|---|
| **Encoder ảnh** | **CLIP** (Radford 2021) | Trích đặc trưng ảnh thô → e_Iv |
| **Encoder text** | **BERT** (Devlin 2019) | Mã hoá mô tả text, caption ảnh, và cả đáp án MLLM sinh ra |
| **MLLM** | **LLaVA-1.5-7B** (Liu 2023), fine-tune bằng **LoRA** | (a) sinh caption cho ảnh; (b) suy luận đoán thực thể thiếu; (c) viết lại text để tạo hard negative |
| **Scoring function** | **RotatE** (Sun 2019) | Chấm điểm bộ ba trong không gian phức |
| **Loss framework** | Self-adversarial negative sampling loss của RotatE + **adversarial training với gradient penalty** (kiểu WGAN-GP) | Huấn luyện |

Nói ngắn: **HFR-MKGC = RotatE + fusion đa mô thức có hướng dẫn bởi quan hệ + LLaVA fine-tuned + adversarial hard negative**.

Tất cả baseline được **tái hiện** với cùng encoder CLIP/BERT để so sánh công bằng (trang 6).

---

## 5. Từng phương pháp: làm gì, input, chạy thế nào, ý nghĩa

### 5.0 Mã hoá mô thức (Modality Encoding, trang 3)
Với mỗi thực thể e và mô thức m ∈ {S, I, T}:
`e_m = P_m(f_m) ∈ R^d`
- f_m: đặc trưng thô từ encoder pre-trained (CLIP cho ảnh, BERT cho text; structural là embedding học tự do).
- P_m: **lớp chiếu tuyến tính học được**, đưa mọi mô thức về cùng chiều d.
- Caption ảnh do MLLM sinh **cũng được xem là một loại text**.

---

### 5.1 Module 1 – Relation-Guided Hierarchical Modal Fusion (RHF)

#### Stage 1: Intra-Visual Modality Fusion (trộn *trong* mô thức ảnh) – Eq. 1, trang 3
- **Vấn đề**: ảnh thô (pixel-level, qua CLIP) và ý nghĩa ảnh (semantic) không nhất quán.
- **Input**: ảnh của thực thể.
- **Chạy**:
  1. Đưa ảnh vào **LLaVA** → sinh **caption mô tả** ảnh.
  2. Caption → BERT → chiếu P_T → `e_Ic` (vector ngữ nghĩa ảnh).
  3. Ảnh → CLIP → chiếu → `e_Iv` (vector thị giác thô).
  4. Trộn bằng **cổng theo độ tương đồng**:

     `e_I = σ(⟨e_Iv, e_Ic⟩) · e_Ic + (1 − σ(⟨e_Iv, e_Ic⟩)) · e_Iv`

     ⟨·,·⟩ là tích vô hướng, σ là sigmoid. Hai vector **càng giống nhau** → tin caption nhiều hơn; càng khác → giữ ảnh thô nhiều hơn.
- **Output**: `e_I ∈ R^d`, biểu diễn ảnh gồm cả low-level visual + high-level semantic.
- **Ý nghĩa**: ảnh một mình rất nhiễu (ví dụ ảnh ca sĩ chỉ có ngoại hình); caption giúp "dịch" ảnh thành ngữ nghĩa mà KG hiểu được.

#### Stage 2: Relation-Guided Cross-Modal Fusion (trộn *giữa* các mô thức theo quan hệ) – Eq. 2–4, trang 3–4
- **Input**: 3 vector của thực thể `H_e = {e_S, e_I, e_T}` và embedding quan hệ `r ∈ R^d` của bộ ba đang xét.
- **Chạy**:
  1. Ma trận tương đồng giữa các mô thức: `A_ij = ⟨e_i, e_j⟩`, i, j ∈ {S, I, T}.
  2. **Điểm tương tác có quan hệ** cho mỗi mô thức m:

     `α_m = ⟨e_m, r⟩ + (1/|M_e|) · Σ_j A_mj`

     Số hạng 1: mô thức m **liên quan đến quan hệ r** bao nhiêu. Số hạng 2: mô thức m **nhất quán với các mô thức khác** bao nhiêu (trung bình tương đồng).
  3. Trọng số bằng softmax có **trừ trung bình** (centered softmax) để ổn định:

     `w_m = exp(α_m − ᾱ) / Σ_j exp(α_j − ᾱ)`, với ᾱ = trung bình các α.
  4. Embedding chung: `H_joint = Σ_m w_m · e_m`.
- **Output**: `H_joint ∈ R^d` cho head, làm tương tự cho tail → `T_joint`.
- **Ý nghĩa**: **cùng một thực thể nhưng quan hệ khác → trọng số khác**. Case study: với *voice type*, text được trọng số cao, ảnh thấp → đoán đúng Mezzo-soprano; fusion không có r xếp đáp án đúng thứ 2 (trang 7).

---

### 5.2 Module 2 – MLLMs-Enhanced Knowledge Reasoning and Scoring (MER)

#### (a) Instruction Fine-Tuning (trang 4)
- Prompt chia 2 phần:
  - **Prom_fix** (cố định): mô tả nhiệm vụ ("cho bộ ba thiếu tail, hãy đoán…") và **định dạng đầu ra** yêu cầu.
  - **Prom_var(h, r)** (biến đổi theo bộ ba): mô tả text của h, **tập láng giềng** của h trong KG, **ảnh** của h, quan hệ r.
- Nhãn giám sát: thực thể tail thật `t`.
- Fine-tune **LLaVA bằng LoRA** → giữ khả năng tổng quát, chỉ học thêm ma trận hạng thấp.

#### (b) Fine-tuned MLLM Reasoning – Eq. 5
`t̂_MLLM = MLLM(Prom_fix ⊕ Prom_var(h, r))`
- **Input**: bộ ba thiếu (h, r, ?) + đa mô thức của h.
- **Output**: **tên thực thể** dự đoán dạng text. Nếu thiếu head thì làm đối xứng với (t, r).
- Sau đó `T_MLLM = BERT(t̂_MLLM)` → vector.

#### (c) Gated Fusion trong không gian phức – Eq. 6–9, trang 4
- **Vấn đề**: MLLM có thể đoán đúng, có thể ảo giác. Không thể tin 100%. Cần một **cổng** quyết định tin bao nhiêu.
- **Chạy**:
  1. Quan hệ r → **số phức đơn vị** (góc pha): `r_C = cos(r/(γ/π)) + i·sin(r/(γ/π))`, γ điều chỉnh miền giá trị. (Đây chính là cách RotatE biểu diễn quan hệ.)
  2. Tách `T_joint` và `T_MLLM` thành phần thực Re(·) và ảo Im(·).
  3. Cổng 2 lớp, tính **từ dự đoán của MLLM**:

     `g = σ(W2 · tanh(W1 · T_MLLM + b1) + b2) ∈ [0,1]^{2d}`, tách `g = [g_re ; g_im]`.
  4. Nội suy:

     `Re(T̃_joint) = (1 − g_re) ∘ Re(T_joint) + g_re ∘ Re(T_MLLM)`
     `Im(T̃_joint) = (1 − g_im) ∘ Im(T_joint) + g_im ∘ Im(T_MLLM)`

     ∘ là nhân từng phần tử.
- **Ý nghĩa**: g gần 1 → tin MLLM; g gần 0 → tin embedding. Cổng học **theo từng chiều** và **theo từng bộ ba** → "adaptive reasoning".

#### (d) RotatE-Based Scoring – Eq. 10
`score(h, r, t) = − ‖ H̃_joint ∘ r_C − T̃_joint ‖`
- Xoay head theo pha của quan hệ; càng gần tail thì khoảng cách càng nhỏ → điểm càng cao (điểm là **âm khoảng cách**).
- Xếp hạng toàn bộ thực thể ứng viên theo điểm này.

> Lưu ý khi đọc: bài viết ký hiệu H̃_joint cho head nhưng chỉ mô tả gate cho tail. Hiểu đơn giản: head là thực thể đã biết nên dùng H_joint từ RHF; tail (thực thể cần đoán) mới cần trộn thêm gợi ý của MLLM.

---

### 5.3 Module 3 – Multi-Modal Negative Sample Optimization (MNO)

Mục tiêu: tạo **hard negative** ở mức đặc trưng đa mô thức (không chỉ thay id thực thể như negative sampling thông thường).

#### (a) MLLM-Based Textual Perturbation (trang 5)
`ẽ_pert^t = P_t(E_text(MLLM(x_orig^t)))`
- Đưa mô tả text gốc của thực thể vào MLLM → sinh câu **cùng nghĩa nhưng khác cách diễn đạt** (paraphrase) → BERT → chiếu.
- Ý nghĩa: negative "gần giống" positive ở mức ngữ nghĩa, ép mô hình phân biệt tinh hơn.

#### (b) Visual Feature Augmentation with Rotation and Noise – Eq. 11
`ẽ_pert^v = ((1 − λ)·I + λ·R) · x_orig^v + ε`
- **R**: ma trận xoay lấy từ **phân rã QR** của ma trận hạng thấp ngẫu nhiên (đảm bảo trực giao).
- **λ ∈ [0,1]**: cường độ xoay, **tăng tuyến tính theo epoch** → curriculum: đầu dễ, sau khó.
- **ε ~ N(0, σ²I)**: nhiễu Gaussian.
- Ý nghĩa: biến dạng nhẹ đặc trưng ảnh, tạo ảnh "giả" khó phát hiện.

#### (c) Ba loại negative đa mô thức
Từ đặc trưng đã nhiễu tạo `(h*, r, t)`, `(h, r, t*)`, `(h*, r, t*)` → M = 3 loại, chấm điểm bằng score(·).

#### (d) Adversarial Training – Eq. 12–14, trang 5
- **Loss KGC (discriminator)** – self-adversarial sigmoid loss (giống RotatE):

  `L_kgc = −(1/2N) Σ_i [ log σ(s_i^+) + Σ_j w_i^j · log σ(−s_i,j^−) ]`

  với `w_i^j = softmax_j(τ · s_i,j^−)` → negative **điểm càng cao (càng khó)** thì **trọng số càng lớn**. τ điều chỉnh độ sắc. Negative gồm cả loại cấu trúc (thay id ngẫu nhiên) và loại đa mô thức ở trên.
- **Loss generator**:

  `L_g = (1/M) Σ_k E[ max(0, c − s_k) ]`

  Generator (bộ tạo nhiễu text/ảnh) bị phạt khi negative loại k có điểm **thấp hơn margin c** → buộc nó sinh mẫu **khó hơn** (điểm cao hơn).
- **Tổng thể**:

  `L_D = L_kgc + μ1 · E[ (‖∇ D(f̂)‖₂ − 1)² ]`   (gradient penalty, f̂ là nội suy giữa embedding thật và giả, ổn định huấn luyện kiểu WGAN-GP)
  `L_g` như trên.

  D = score(·) là discriminator. Tối ưu xen kẽ D và G.
- **Ý nghĩa**: mô hình học ranh giới đúng/sai sắc hơn, robust hơn với nhiễu mô thức; ablation cho thấy bỏ adversarial (w/o MNO) giảm MRR từ 38.62 → 34.01.

---

### 5.4 Luồng dữ liệu tổng thể (data flow – cần cho slide)

```
Ảnh h ──CLIP──► e_Iv ─┐
Ảnh h ──LLaVA caption──BERT──► e_Ic ─┴─[Stage 1: gate similarity]──► e_I ─┐
Text h ──BERT──► e_T ─────────────────────────────────────────────────────┤
Embedding cấu trúc h ──► e_S ─────────────────────────────────────────────┤
Quan hệ r ──────────────────────────────────────────►[Stage 2: α_m, w_m]──┴──► H_joint
                                                                                 │
(h, r, ?) + text + láng giềng + ảnh ──► Prompt ──► LLaVA (LoRA) ──► t̂_MLLM ──BERT──► T_MLLM
                                                                                 │
Ứng viên t ──(RHF như trên)──► T_joint ──[Gate g từ T_MLLM]──► T̃_joint          │
                                                                                 ▼
                         score = −‖ H_joint ∘ r_C − T̃_joint ‖  ──► xếp hạng ứng viên
                                                                                 ▲
Hard negative: MLLM paraphrase text, xoay+nhiễu ảnh ──► (h*,r,t),(h,r,t*),(h*,r,t*) ─┘
Huấn luyện: L_D (KGC + gradient penalty)  ⇄  L_g (generator)
```

### 5.5 Thiết lập huấn luyện (Implementation Details, trang 5–6)
- PyTorch, Ubuntu, **2 × NVIDIA RTX A6000 (48GB)**.
- **250 epoch**, tối ưu **Adam**.
- Chiều embedding d ∈ {128, 256, 512}; số negative/bộ ba ∈ {32, 64, 128}; learning rate ∈ {1e-5, 1e-4, 1e-3}; batch size ∈ {128, 256, 1024}.
- Baseline được tái hiện với cùng encoder để so sánh công bằng.

---

## 6. Đánh giá: metric và ý nghĩa

Bài toán đánh giá là **link prediction**: với mỗi bộ ba test, che tail (và che head), chấm điểm **toàn bộ thực thể** làm ứng viên, xem đáp án thật đứng hạng mấy.

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| **MRR** (Mean Reciprocal Rank) | trung bình của `1 / rank` của đáp án đúng | Đáp án đúng hạng 1 → 1.0; hạng 2 → 0.5; hạng 10 → 0.1. Nhạy với thứ hạng cao, phản ánh **chất lượng xếp hạng tổng thể**. Càng cao càng tốt. |
| **Hits@1** | % bộ ba mà đáp án đúng đứng **hạng 1** | Độ chính xác "đoán trúng ngay". Khó nhất. |
| **Hits@3** | % đáp án đúng nằm **top 3** | Mức trung gian. |
| **Hits@10** | % đáp án đúng nằm **top 10** | Dễ nhất, đo khả năng "không bỏ sót" ứng viên. |

Tất cả tính theo %, càng cao càng tốt. Bài dùng chuẩn **filtered setting** (ngầm định theo các công trình trước, bài không nói rõ).

Với baseline LLM thuần (LLaMA, LLaVA), MLLM chỉ sinh **danh sách 10 ứng viên**, không chấm điểm hết mọi thực thể → **MRR không tính được** (ký hiệu "—" trong Table 1).

### 6.1 Kết quả chính (Table 1, trang 6, đơn vị %)

| Model | MKG-W MRR | H@10 | H@3 | H@1 | MKG-Y MRR | H@10 | H@3 | H@1 | DB15K MRR | H@10 | H@3 | H@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RotatE (uni-modal) | 32.92 | 44.23 | 35.91 | 26.68 | 36.67 | 41.11 | 35.91 | 33.92 | 32.07 | 51.40 | 38.57 | 21.23 |
| RSME | 33.41 | 44.31 | 35.80 | 27.59 | 38.03 | 44.02 | 35.80 | 34.28 | 33.72 | 50.39 | 39.23 | **24.45** |
| MyGO | 33.40 | 44.36 | 35.38 | 27.52 | 30.29 | 42.17 | 35.38 | 23.75 | 32.25 | 48.23 | 35.49 | 24.19 |
| AdaMF-MAT | 32.09 | 45.78 | 35.54 | 24.48 | 37.78 | 46.11 | 35.54 | 32.65 | 32.08 | 52.42 | 40.41 | 19.81 |
| APKGC | 32.67 | 46.20 | 35.83 | 25.24 | 31.48 | 40.68 | 35.83 | 26.17 | 32.91 | 48.91 | 37.15 | 24.39 |
| **NativeE** (best baseline) | 36.79 | 50.28 | 40.47 | 29.36 | 38.34 | 46.24 | 40.47 | 33.49 | 33.59 | 53.77 | 40.67 | 22.11 |
| LLaMA-2-7B (prompt only) | — | 10.20 | 8.73 | 5.62 | — | 2.29 | 2.03 | 1.35 | — | 14.14 | 11.87 | 7.15 |
| LLaVA-1.5-7B (prompt only) | — | 26.42 | 22.29 | 15.02 | — | 3.58 | 2.80 | 1.64 | — | 15.89 | 13.15 | 8.32 |
| **HFR-MKGC** | **38.62** | **51.66** | **42.12** | **31.37** | **39.00** | **47.31** | **42.12** | **34.41** | **35.20** | **56.41** | **43.25** | 23.04 |

(Bảng đầy đủ 14 baseline trong PDF; ở đây giữ các baseline đáng chú ý. TransE, DistMult, ComplEx, IKRL, SNAG, DHNS đều thấp hơn.)

**Nhận xét**:
- HFR-MKGC **đứng đầu ở 11/12 ô**. Duy nhất **Hits@1 trên DB15K** (23.04) thua RSME (24.45) và APKGC (24.39), MyGO (24.19) — bài báo nói "SOTA trên tất cả metric" nhưng bảng cho thấy ngoại lệ này; nên nêu trung thực trong seminar.
- Trên MKG-W: hơn NativeE **+1.83 MRR**, +2.01 H@1, +1.65 H@3, +1.38 H@10.
- Cải thiện trên MKG-Y **nhỏ nhất** (+0.66 MRR) — tập này ít quan hệ (28), cấu trúc đã đủ mạnh, đa mô thức thêm ít giá trị.
- **LLM/MLLM thuần rất kém** (Hits@10 ≤ 26%): LLM một mình không thay được scoring có cấu trúc. LLaVA (có ảnh) > LLaMA (không ảnh) ở mọi tập → ảnh có ích.

### 6.2 Ablation trên MKG-W (Table 3, trang 7, %)

| Cấu hình | MRR | H@10 | H@3 | H@1 | Bỏ gì |
|---|---|---|---|---|---|
| w/o S | 25.67 | 39.42 | 28.81 | 18.02 | bỏ mô thức cấu trúc |
| w/o T | 29.67 | 45.61 | 34.92 | 21.25 | bỏ text |
| w/o V | 31.16 | 47.39 | 36.10 | 22.09 | bỏ ảnh |
| w/o VT | 31.78 | 47.33 | 36.41 | 23.22 | bỏ caption ảnh do MLLM sinh |
| w/o RHF | 31.98 | 50.25 | 39.14 | 20.80 | thay fusion có quan hệ bằng **trung bình** |
| w/o MER | 33.52 | 50.61 | 39.70 | 23.42 | bỏ suy luận MLLM |
| w/o PTI | 34.59 | 50.95 | 40.03 | 25.12 | thay perturbation text/ảnh bằng **nhiễu ngẫu nhiên** |
| w/o MNO | 34.01 | 50.98 | 40.26 | 24.13 | bỏ adversarial training |
| **Full** | **38.62** | **51.66** | **42.12** | **31.37** | |

**Đọc bảng**:
- **Cấu trúc (S) quan trọng nhất**: bỏ S mất ~13 điểm MRR. Text quan trọng hơn ảnh (bỏ T mất 9, bỏ V mất 7.5).
- **RHF là module đóng góp lớn nhất** trong 4 module (mất 6.6 MRR, H@1 rớt từ 31.4 → 20.8) → relation-guided fusion **chủ yếu giúp xếp đúng hạng 1**, còn H@10 gần như không đổi.
- MER (MLLM reasoning) đóng góp ~5 MRR.
- Hard negative có nghĩa (PTI) và adversarial (MNO) mỗi thứ ~4–4.6 MRR.
- Caption ảnh (VT) có ích: bỏ mất ~6.8 MRR.

### 6.3 Độ nhạy siêu tham số (Figure 3, trang 7)
- Chiều lớn (512) cần **nhiều negative** (128) mới phát huy; chiều vừa (256) tốt nhất với 64 negative; chiều nhỏ (128) gần như không ảnh hưởng bởi số negative.
- Thiếu negative thì 256D có thể **hơn** 512D.
- **MRR và Hits@1 nhạy** với siêu tham số, **Hits@10 ổn định**.

### 6.4 Case study (Figure 4, trang 7)
Query: **(Gwen Stefani, voice type, ?)**, đáp án **Mezzo-soprano**.
- Không có RHF (trọng số ngẫu nhiên): đáp án đúng xếp **hạng 2**.
- Có RHF: ảnh chỉ chứa ngoại hình, không nói gì về giọng; text nói cô là ca sĩ → RHF **tăng trọng số text, giảm trọng số ảnh** → đúng **hạng 1**.
→ Ví dụ input–output hoàn hảo cho slide.

---

## 7. Kết luận

**Tác giả kết luận** (trang 7): HFR-MKGC đưa ra (1) fusion phân cấp có hướng dẫn bởi quan hệ, (2) MLLM fine-tuned suy luận theo hướng dẫn cho từng bộ ba, (3) hard negative đa mô thức + adversarial training. SOTA trên 3 benchmark. Hướng tương lai: khai thác MLLM **chính xác hơn** cho biểu diễn và suy luận tri thức đa mô thức.

**Nhận xét / hạn chế (tự đánh giá, dùng cho phần thảo luận & Q&A)**:
1. **Chi phí tính toán lớn**: cần chạy LLaVA-7B cho *mỗi* ảnh (caption), *mỗi* query (reasoning) và *mỗi* text (paraphrase). Bài **không báo cáo thời gian huấn luyện/suy luận** hay so sánh chi phí với baseline.
2. **Không có code công khai** → khó tái lập. Nhiều baseline được tác giả tự tái hiện.
3. **Cải thiện không đồng đều**: MKG-Y chỉ +0.66 MRR; thua Hits@1 trên DB15K. Bài vẫn tuyên bố "SOTA ở mọi metric".
4. **Ablation chỉ trên MKG-W**, không kiểm tra trên hai tập còn lại.
5. **MLLM chỉ sinh 1 đáp án** t̂_MLLM rồi BERT-encode; nếu MLLM ảo giác một tên không tồn tại, gate phải học cách bỏ qua — bài không phân tích tỉ lệ MLLM đoán đúng/sai hay gate hoạt động thế nào.
6. **Rò rỉ tri thức tiềm ẩn**: LLaVA/BERT pre-trained trên web có thể đã "biết" sự thật trong Wikidata/DBpedia; bài không thảo luận.
7. Chi tiết prompt, số epoch fine-tune LoRA, cách chọn láng giềng đưa vào prompt **không được mô tả**.
8. Ký hiệu trong bài đôi chỗ chưa nhất quán (H̃_joint chưa định nghĩa; hai tài liệu Chen 2025a/2025b cùng tên "Noise-powered…").

**Câu hỏi Q&A có thể gặp**:
- Tại sao dùng RotatE mà không phải scoring khác? → RotatE mạnh ở quan hệ đối xứng/nghịch đảo/hợp thành; và biểu diễn phức thuận tiện cho gate thực/ảo.
- Gate g tính từ T_MLLM chứ không từ cả T_joint, vì sao? → Để mô hình học "độ tin cậy" của chính đáp án MLLM.
- Hard negative đa mô thức khác gì negative thay id? → Negative thay id tạo bộ ba sai về cấu trúc; negative đa mô thức giữ nguyên thực thể nhưng **bóp méo đặc trưng**, ép mô hình học biểu diễn robust.
- Nếu hai node không có quan hệ thì sao, có đặt ngưỡng không? → Bài **không đặt ngưỡng**. Đánh giá chỉ hỏi bộ ba biết chắc là đúng rồi xem đáp án đứng hạng mấy (link prediction), nên model chỉ cần xếp hạng, không cần quyết định có/không. Muốn dùng thật phải tự thêm ngưỡng trên validation (triple classification), thường theo từng quan hệ vì thang điểm RotatE không chuẩn hoá.
- Model có thêm được thực thể mới không? → Không, đây là **transductive**: mỗi node là một vector tra bảng học lúc train. Node mới phải train lại. Bài toán node mới là inductive KGC (CATS trong related work), bài này không giải quyết.
- Vì sao LLM thuần kém? → Không có scoring trên toàn bộ tập thực thể, chỉ sinh 10 ứng viên; không biết ranh giới tập thực thể; ảo giác.

---

## 8. Bản đồ trang PDF để tra nhanh

| Nội dung | Trang |
|---|---|
| Abstract, Introduction, 2 thách thức | 1 |
| Figure 1 (3 paradigm), Contributions, Related Work | 2 |
| Figure 2 (kiến trúc), Task Definition, Encoding, Stage 1, đầu Stage 2 | 3 |
| Eq. 2–10: Stage 2, Instruction FT, Reasoning, Gated Fusion, RotatE score | 4 |
| Eq. 11–14: Perturbation, Augmentation, Adversarial loss; Datasets, Metrics, Baselines | 5 |
| Table 1 (kết quả), Table 2 (dataset), Main Results, Ablation mô tả | 6 |
| Table 3 (ablation), Figure 3 (siêu tham số), Figure 4 (case study), Conclusion | 7 |
| References | 8–9 |
