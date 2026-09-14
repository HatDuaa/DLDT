# Tóm tắt bài báo HFR-MKGC (bản dễ hiểu)

**HFR-MKGC: Hierarchical Fusion Reasoning with MLLMs for Multi-modal Knowledge Graph Completion**
D. Wang, J. Du, Z. Xue, M. Liang, G. Ye, Y. Shao, H. Li — AAAI 2026, vol. 40, tr. 15815–15823.
Nguồn: [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/38613) · PDF gốc trong `original/paper/`.
Mọi số liệu lấy trực tiếp từ bài báo, có ghi số trang PDF (tr. 1 = trang đầu bài báo). Tác giả **không công khai code**.

---

## Đọc trong 1 phút

- **Bài toán:** đồ thị tri thức (KG) luôn thiếu cạnh. Cho bộ ba thiếu `(Gwen Stefani, voice type, ?)`, phải xếp hạng toàn bộ thực thể để tìm đáp án `Mezzo-soprano`. Ở đây mỗi thực thể có thêm **ảnh** và **mô tả text**, gọi là multi-modal KGC (MMKGC).
- **Vấn đề:** các phương pháp trước trộn ba nguồn thông tin (cấu trúc, ảnh, text) với **trọng số cố định** cho mọi quan hệ. Nhưng với quan hệ *voice type* ảnh gần như vô dụng, với quan hệ *màu lông* ảnh lại là tất cả. Ngoài ra, LLM dùng cho KGC mới chỉ đọc text, chưa dùng ảnh, và không biết "nên tin LLM bao nhiêu".
- **Giải pháp:** HFR-MKGC = **RotatE** (hàm chấm điểm) + **trộn đa mô thức có trọng số theo quan hệ** + **LLaVA fine-tune đoán thẳng đáp án rồi trộn qua cổng** + **hard negative đa mô thức, huấn luyện đối kháng**.
- **Kết quả:** đứng đầu 11/12 ô trên 3 benchmark so với 14 baseline. Ngoại lệ: Hits@1 trên DB15K thua RSME.
- **Ví dụ minh hoạ:** query (Gwen Stefani, voice type, ?). Không có trộn theo quan hệ, đáp án đúng xếp hạng 2. Có trộn theo quan hệ, text được ưu tiên, ảnh bị giảm, đáp án lên hạng 1 (Figure 4, tr. 7).

---

## 0. Kiến thức nền (chỉ những gì cần để đọc bài)

| Thuật ngữ | Hiểu ngắn gọn |
|---|---|
| **KG, bộ ba (h, r, t)** | Sự thật lưu dạng (head, relation, tail). Ví dụ (Gwen Stefani, voice type, Mezzo-soprano). |
| **KGC / link prediction** | Cho (h, r, ?) hoặc (?, r, t), **xếp hạng mọi thực thể** làm ứng viên. Không có ngưỡng đúng/sai, chỉ có thứ hạng. |
| **MMKG / MMKGC** | KG mà mỗi thực thể có thêm ảnh và mô tả text. MMKGC là KGC tận dụng thêm hai nguồn này. |
| **Mô thức (modality)** | Một loại thông tin của thực thể. Bài dùng 3 loại: **S** cấu trúc, **I** (hay V) ảnh, **T** text. |
| **Embedding tra bảng** | Mỗi thực thể là một vector học từ đầu, kiểu tra bảng. Không có message passing như GNN. |
| **Scoring function** | Hàm cho điểm "độ hợp lý" của một bộ ba. Xếp hạng ứng viên theo điểm. |
| **RotatE** | Embedding là **số phức**, quan hệ là **phép xoay** góc θ. Bộ ba đúng khi xoay h theo r thì rơi gần t. Điểm = âm khoảng cách. |
| **Negative sampling** | Tạo bộ ba sai để mô hình học phân biệt. **Hard negative** = sai nhưng trông rất giống đúng, dạy mô hình tinh hơn. |
| **CLIP / BERT** | Encoder đóng băng. CLIP biến ảnh thành vector, BERT biến câu thành vector. Bài chỉ học thêm lớp chiếu phía sau. |
| **MLLM, LLaVA, LoRA** | MLLM = LLM nhận cả ảnh lẫn text. Bài dùng LLaVA-1.5-7B. LoRA = fine-tune bằng vài ma trận hạng thấp, rẻ, giữ kiến thức gốc. |
| **Adversarial training** | Generator sinh mẫu sai khó, discriminator (ở đây là hàm điểm) cố phân biệt. Gradient penalty (kiểu WGAN-GP) để huấn luyện ổn định. |

---

## 1. Vấn đề bài báo giải quyết (tr. 1–2)

### Bối cảnh: "trộn mô thức" nghĩa là gì?

Lấy thực thể **Gwen Stefani** làm ví dụ xuyên suốt. Trong MMKG, cô có ba nguồn thông tin:

| Mô thức | Nội dung thực tế | Encoder → vector |
|---|---|---|
| **S** cấu trúc | các cạnh đã biết: (Gwen Stefani, member of, No Doubt), (Gwen Stefani, place of birth, Anaheim) | embedding học từ đầu → `e_S` |
| **T** text | "Gwen Renée Stefani (born October 3, 1969) is an American singer, songwriter, actress…" | BERT → `e_T` |
| **I** ảnh | một tấm ảnh người phụ nữ tóc vàng đang biểu diễn | CLIP → `e_I` |

Mô hình không thể chấm điểm bằng ba vector rời rạc, nên phải **trộn** chúng thành một vector duy nhất đại diện cho Gwen Stefani:

```
e_Gwen = w_S · e_S + w_I · e_I + w_T · e_T        (w là trọng số, cộng lại bằng 1)
```

Câu hỏi then chốt của bài báo là: **w lấy ở đâu ra?**

### Điểm yếu 1: trộn mô thức "mù quan hệ" (relation-agnostic)

Các phương pháp trước chọn w **một lần rồi dùng cho mọi câu hỏi**. Ví dụ RSME học được rằng ảnh nhìn chung khá hữu ích trên tập dữ liệu này, nên cố định `w_S = 0.3, w_I = 0.5, w_T = 0.2` cho Gwen Stefani, bất kể đang hỏi gì.

Bây giờ đặt hai câu hỏi khác nhau về **cùng một thực thể**:

| Query | Mô thức nào thật sự chứa câu trả lời? | Mô thức nào là nhiễu? |
|---|---|---|
| (Gwen Stefani, **voice type**, ?) → Mezzo-soprano | **Text** ("singer") gợi ý đây là giọng ca sĩ nữ | **Ảnh** chỉ thấy ngoại hình, không nói gì về giọng |
| (Gwen Stefani, **hair color**, ?) → Blonde | **Ảnh** thấy rõ tóc vàng | **Text** không nhắc màu tóc |

Với w cố định `(0.3, 0.5, 0.2)`:
- Hỏi *hair color*: ảnh được 0.5, ổn, đoán đúng.
- Hỏi *voice type*: ảnh vẫn được 0.5 dù vô dụng. Vector ảnh "kéo" e_Gwen về phía những thực thể trông giống một buổi biểu diễn, mô hình chấm **Double bass** (một nhạc cụ) cao hơn Mezzo-soprano. Đây chính là lỗi trong case study của bài (Figure 4, tr. 7): đáp án đúng chỉ xếp hạng 2.

Tác giả gọi kiểu này là **relation-agnostic**: trọng số không biết quan hệ đang hỏi là gì. Điều bài báo muốn là w phải **đổi theo quan hệ r**:

```
hỏi voice type  →  w = (0.35, 0.15, 0.50)   ảnh giảm, text tăng
hỏi hair color  →  w = (0.20, 0.60, 0.20)   ảnh tăng
```

(Các con số w ở trên là minh hoạ do nhóm đặt, bài báo không in số cụ thể. Chỉ Figure 4 vẽ cột trọng số t/i/s.)

Một ví dụ khác cùng loại, lấy từ Figure 1 của bài: ảnh chó Golden Retriever ngậm cây gậy. Hỏi *màu lông* thì ảnh là tất cả; hỏi *quốc gia xuất xứ giống chó* thì ảnh vô dụng, text mới có "Scotland".

### Điểm yếu 2: LLM cho KGC còn cứng

Hướng thứ hai đang thịnh hành là bỏ hẳn hàm chấm điểm, viết KG thành prompt rồi hỏi LLM:

```
Prompt: "Gwen Stefani is a member of No Doubt, born in Anaheim.
         Description: American singer, songwriter, actress.
         Question: What is the voice type of Gwen Stefani?"
LLM:    "Soprano"
```

Cách này gặp ba vấn đề cụ thể:

1. **Không có ảnh.** Prompt chỉ có text và cấu trúc. Hỏi *hair color* thì LLM không có gì để trả lời, trong khi ảnh trả lời được ngay. (LLaMA-2 trong Table 1 là ví dụ: không ảnh, Hits@10 chỉ 10.2% trên MKG-W.)
2. **Trộn tĩnh, không biết "tin LLM bao nhiêu".** Nếu hệ thống lấy đáp án LLM rồi ghép vào embedding với một tỉ lệ cố định, thì khi LLM ảo giác ("Soprano" thay vì "Mezzo-soprano") hệ thống vẫn tin y như khi LLM đúng. Mỗi bộ ba cần một mức tin khác nhau: bộ ba dễ, LLM thường đúng, tin nhiều; bộ ba hiếm, LLM hay bịa, tin ít.
3. **LLM không xếp hạng được toàn bộ 15 000 thực thể.** Nó chỉ sinh vài tên (bài để LLM sinh 10 ứng viên). Không có điểm cho mọi ứng viên nên không tính được MRR, và nếu đáp án đúng không nằm trong 10 tên đó thì coi như trượt. Kết quả: LLaVA thuần chỉ đạt Hits@10 = 26.4% trên MKG-W, trong khi HFR-MKGC đạt 51.7%.

### Tóm lại hai điểm yếu dẫn tới hai ý tưởng chính

| Điểm yếu | Ý tưởng khắc phục | Module |
|---|---|---|
| Trọng số trộn không đổi theo quan hệ | Tính w từ vector quan hệ r, mỗi query một bộ w | RHF (mục 3) |
| LLM không có ảnh, không biết tin bao nhiêu, không xếp hạng được | Dùng **MLLM** (có ảnh), đáp án của nó được **cổng g** quyết định tin bao nhiêu, rồi vẫn chấm điểm bằng RotatE trên **toàn bộ** thực thể | MER (mục 4) |

**Ba paradigm** (Figure 1, tr. 2):
- (a) MMKGC truyền thống: encoder → fusion → score.
- (b) LLM-based KGC: KG → prompt → LLM → đáp án.
- (c) HFR-MKGC: kết hợp cả hai, trộn có hướng dẫn bởi quan hệ, cộng thêm hard negative.

**Đóng góp tác giả tự nêu** (tr. 2): framework mới với trộn phân cấp có quan hệ dẫn hướng; chiến lược MLLM suy luận theo từng bộ ba rồi tích hợp vào chấm điểm; SOTA trên 3 benchmark so với 14 baseline.

---

## 2. Bức tranh tổng thể

Mỗi thực thể có ba vector (S, I, T). Với một query (h, r, ?), mô hình làm ba việc:

```
                ┌──────────── Module 1: RHF ────────────┐
Ảnh h ─CLIP─► e_Iv ─┐                                   │
Ảnh h ─LLaVA caption─BERT─► e_Ic ─┴─[cổng tương đồng]─► e_I ─┐
Text h ─BERT─► e_T ──────────────────────────────────────────┤
Cấu trúc h ─► e_S ───────────────────────────────────────────┤
Quan hệ r ─────────────────────────────► [trọng số theo r] ──┴─► H_joint
                ┌──────────── Module 2: MER ────────────┐
(h, r, ?) + text + láng giềng + ảnh ─► prompt ─► LLaVA (LoRA) ─► tên đáp án ─BERT─► T_MLLM
Ứng viên t ─(RHF)─► T_joint ─[cổng g tính từ T_MLLM]─► T̃_joint
                              score = −‖ H_joint ∘ r_C − T̃_joint ‖  ─► xếp hạng
                ┌──────────── Module 3: MNO ────────────┐
Hard negative: LLaVA viết lại text, xoay + nhiễu vector ảnh ─► (h*,r,t), (h,r,t*), (h*,r,t*)
Huấn luyện: L_D (KGC + gradient penalty) đấu với L_g (generator)
```

Ba module (Figure 2, tr. 3):
1. **RHF** – Relation-Guided Hierarchical Modal Fusion: trộn phân cấp, trọng số theo quan hệ.
2. **MER** – MLLMs-Enhanced Knowledge Reasoning and Scoring: LLaVA đoán đáp án, trộn qua cổng, chấm điểm RotatE.
3. **MNO** – Multi-Modal Negative Sample Optimization: hard negative đa mô thức, huấn luyện đối kháng.

**Mô hình nền được dùng** (tr. 3–5):

| Thành phần | Mô hình | Vai trò |
|---|---|---|
| Encoder ảnh | CLIP | ảnh thô → vector |
| Encoder text | BERT | mô tả text, caption ảnh, và cả đáp án LLaVA sinh ra → vector |
| MLLM | LLaVA-1.5-7B + LoRA | sinh caption ảnh; đoán thực thể thiếu; viết lại text tạo negative |
| Scoring | RotatE | chấm điểm trong không gian phức |
| Loss | self-adversarial loss của RotatE + adversarial với gradient penalty | huấn luyện |

Tất cả baseline được tái hiện với cùng encoder CLIP/BERT để so sánh công bằng (tr. 6).

---

## 3. Module 1 – RHF: trộn phân cấp có quan hệ dẫn hướng (tr. 3–4)

### 3.0 Mã hoá mô thức

Mỗi thực thể e, mỗi mô thức m ∈ {S, I, T}: `e_m = P_m(f_m) ∈ R^d`.
- `f_m`: đặc trưng thô từ encoder (CLIP cho ảnh, BERT cho text; S là embedding học tự do).
- `P_m`: lớp chiếu tuyến tính học được, đưa mọi mô thức về cùng chiều d.

### 3.1 Tầng 1: trộn *bên trong* mô thức ảnh (Eq. 1, tr. 3)

**Tại sao cần?** Ảnh qua CLIP chỉ cho "pixel-level": màu, hình dáng. Nhưng KG cần ngữ nghĩa: đây là ai, làm gì. Tác giả dùng LLaVA **viết caption** cho ảnh, coi caption như một dạng text, để "dịch" ảnh sang ngôn ngữ mà KG hiểu.

**Chạy thế nào:**
1. Ảnh → CLIP → chiếu → `e_Iv` (vector thị giác thô).
2. Ảnh → LLaVA sinh caption → BERT → chiếu → `e_Ic` (vector ngữ nghĩa ảnh).
3. Trộn bằng cổng:

```
e_I = σ(⟨e_Iv, e_Ic⟩) · e_Ic  +  (1 − σ(⟨e_Iv, e_Ic⟩)) · e_Iv
```

| Thành phần | Nghĩa |
|---|---|
| `⟨e_Iv, e_Ic⟩` | tích vô hướng, đo ảnh thô và caption "giống nhau" bao nhiêu |
| `σ(·)` | sigmoid, đưa về [0, 1] thành trọng số |
| trọng số cao → tin caption | hai vector nhất quán, caption đáng tin |
| trọng số thấp → giữ ảnh thô | caption lệch với ảnh, giữ thông tin gốc |

**Output:** `e_I ∈ R^d`, vừa có thị giác thô vừa có ngữ nghĩa.

### 3.2 Tầng 2: trộn *giữa* ba mô thức theo quan hệ (Eq. 2–4, tr. 3–4)

**Ý tưởng cốt lõi của bài:** trọng số trộn S/I/T phải phụ thuộc vào quan hệ r đang hỏi.

**Input:** ba vector `{e_S, e_I, e_T}` của thực thể và vector quan hệ `r ∈ R^d`.

**Bước 1.** Ma trận tương đồng giữa các mô thức: `A_ij = ⟨e_i, e_j⟩`.

**Bước 2.** Điểm cho mỗi mô thức m:

```
α_m = ⟨e_m, r⟩  +  (1/|M_e|) · Σ_j A_mj
```

| Thành phần | Nghĩa |
|---|---|
| `⟨e_m, r⟩` | mô thức m **liên quan đến quan hệ r** bao nhiêu (số hạng quan trọng nhất) |
| `(1/|M_e|) Σ_j A_mj` | mô thức m **nhất quán với các mô thức khác** bao nhiêu (trung bình tương đồng), giúp loại mô thức lạc lõng |

**Bước 3.** Softmax có trừ trung bình để ổn định số:

```
w_m = exp(α_m − ᾱ) / Σ_j exp(α_j − ᾱ),   ᾱ = trung bình các α
```

**Bước 4.** Embedding chung: `H_joint = Σ_m w_m · e_m`. Làm tương tự cho tail được `T_joint`.

**Điểm mấu chốt:** cùng một thực thể, hỏi quan hệ khác thì w khác. Case study tr. 7: với *voice type*, w_T cao, w_I thấp → đúng hạng 1. Trộn không có r → đáp án đúng chỉ hạng 2.

---

## 4. Module 2 – MER: LLaVA suy luận rồi trộn qua cổng (tr. 4)

### 4.1 Fine-tune theo hướng dẫn (instruction fine-tuning)

Prompt gồm hai phần:
- **Prom_fix** (cố định cho mọi mẫu): mô tả nhiệm vụ ("bạn là chuyên gia suy luận KG, cho bộ ba thiếu tail, hãy đoán…") và định dạng đầu ra.
- **Prom_var(h, r)** (thay đổi theo bộ ba): mô tả text của h, **tập láng giềng** của h trong KG, **ảnh** của h, quan hệ r.

Nhãn giám sát là tail thật `t`. Fine-tune LLaVA bằng **LoRA** để giữ khả năng tổng quát.

### 4.2 Suy luận (Eq. 5)

```
t̂_MLLM = MLLM(Prom_fix ⊕ Prom_var(h, r))
```

LLaVA trả ra **tên thực thể** dạng text. Sau đó `T_MLLM = BERT(t̂_MLLM)` thành vector. Thiếu head thì làm đối xứng với (t, r).

### 4.3 Cổng trộn trong không gian phức (Eq. 6–9)

**Tại sao cần cổng?** LLaVA có thể đoán đúng, cũng có thể ảo giác. Không thể tin 100%. Cần một cổng học "tin LLM bao nhiêu" cho từng bộ ba, từng chiều.

**Bước 1.** Quan hệ r thành số phức đơn vị (chính là cách RotatE biểu diễn quan hệ):

```
r_C = cos(r / (γ/π)) + i · sin(r / (γ/π))
```
γ điều chỉnh miền giá trị pha.

**Bước 2.** Tách `T_joint` và `T_MLLM` thành phần thực Re(·) và phần ảo Im(·).

**Bước 3.** Cổng hai lớp, tính **từ dự đoán của LLaVA**:

```
g = σ(W2 · tanh(W1 · T_MLLM + b1) + b2) ∈ [0,1]^{2d},   g = [g_re ; g_im]
```

**Bước 4.** Nội suy từng phần:

```
Re(T̃_joint) = (1 − g_re) ∘ Re(T_joint) + g_re ∘ Re(T_MLLM)
Im(T̃_joint) = (1 − g_im) ∘ Im(T_joint) + g_im ∘ Im(T_MLLM)
```

| Thành phần | Nghĩa |
|---|---|
| `g` gần 1 | tin LLaVA, lấy nhiều từ T_MLLM |
| `g` gần 0 | tin embedding, giữ T_joint |
| `∘` | nhân từng phần tử, nên cổng học **theo từng chiều** |
| tính từ T_MLLM | cổng học "độ tin cậy" của chính đáp án LLaVA |

### 4.4 Chấm điểm RotatE (Eq. 10)

```
score(h, r, t) = − ‖ H̃_joint ∘ r_C − T̃_joint ‖
```

Xoay head theo pha quan hệ, càng gần tail thì khoảng cách càng nhỏ, điểm càng cao. Xếp hạng mọi ứng viên theo điểm này.

> Lưu ý khi đọc: bài ký hiệu H̃_joint cho head nhưng chỉ mô tả cổng cho tail. Hiểu đơn giản: head đã biết nên dùng H_joint từ RHF; tail (cần đoán) mới trộn thêm gợi ý của LLaVA.

---

## 5. Module 3 – MNO: hard negative đa mô thức và huấn luyện đối kháng (tr. 4–5)

**Khác gì negative thường?** Negative thường thay id thực thể, tạo bộ ba sai về cấu trúc. Bài này giữ nguyên thực thể nhưng **bóp méo đặc trưng** ảnh và text, tạo mẫu "gần giống thật" ép mô hình học biểu diễn bền hơn.

### 5.1 Nhiễu text bằng LLaVA

```
ẽ_pert^t = P_t(BERT(MLLM(x_orig^t)))
```
Đưa mô tả text gốc vào LLaVA → sinh câu **cùng nghĩa, khác cách diễn đạt** (paraphrase) → BERT → chiếu.

### 5.2 Nhiễu ảnh bằng xoay và nhiễu Gaussian (Eq. 11)

```
ẽ_pert^v = ((1 − λ)·I + λ·R) · x_orig^v + ε
```

| Thành phần | Nghĩa |
|---|---|
| `R` | ma trận xoay, lấy từ **phân rã QR** của ma trận hạng thấp ngẫu nhiên (đảm bảo trực giao) |
| `λ ∈ [0, 1]` | cường độ xoay, **tăng tuyến tính theo epoch** → curriculum: đầu dễ, sau khó |
| `ε ~ N(0, σ²I)` | nhiễu Gaussian |

### 5.3 Ba loại negative

Từ đặc trưng đã nhiễu tạo `(h*, r, t)`, `(h, r, t*)`, `(h*, r, t*)` → M = 3 loại, chấm bằng score(·).

### 5.4 Huấn luyện đối kháng (Eq. 12–14, tr. 5)

**Loss KGC (phía discriminator):**

```
L_kgc = −(1/2N) Σ_i [ log σ(s_i^+) + Σ_j w_i^j · log σ(−s_i,j^−) ]
w_i^j = softmax_j(τ · s_i,j^−)
```

| Thành phần | Nghĩa |
|---|---|
| `s_i^+` | điểm bộ ba đúng, muốn cao |
| `s_i,j^−` | điểm negative thứ j, muốn thấp |
| `w_i^j` | negative **điểm càng cao (càng khó)** thì trọng số càng lớn (self-adversarial của RotatE); τ chỉnh độ sắc |
| negative | gồm cả loại thay id ngẫu nhiên và loại đa mô thức ở trên |

**Loss generator:**

```
L_g = (1/M) Σ_k E[ max(0, c − s_k) ]
```
Generator (bộ tạo nhiễu text/ảnh) bị phạt khi negative loại k có điểm **thấp hơn margin c** → buộc sinh mẫu khó hơn.

**Tổng thể:**

```
L_D = L_kgc + μ1 · E[ (‖∇ D(f̂)‖₂ − 1)² ]     (gradient penalty)
L_g như trên
```
D = score(·) là discriminator; f̂ là nội suy giữa embedding thật và giả (kiểu WGAN-GP). Tối ưu xen kẽ D và G.

**Bằng chứng nó có ích:** bỏ adversarial (w/o MNO) làm MRR trên MKG-W giảm từ 38.62 xuống 34.01.

---

## 6. Thiết lập huấn luyện (tr. 5–6)

- PyTorch, 2 × NVIDIA RTX A6000 (48GB), 250 epoch, Adam.
- Chiều embedding d ∈ {128, 256, 512}; số negative/bộ ba ∈ {32, 64, 128}; learning rate ∈ {1e-5, 1e-4, 1e-3}; batch ∈ {128, 256, 1024}.
- Chi tiết prompt, số epoch LoRA, cách chọn láng giềng đưa vào prompt **không được mô tả**.

---

## 7. Dữ liệu (Table 2, tr. 6)

| Dataset | Nguồn | #Entity | #Relation | #Train | #Valid | #Test | #Image | #Text |
|---|---|---|---|---|---|---|---|---|
| **MKG-W** | Xu et al. 2022 (Wikidata) | 15 000 | 169 | 34 196 | 4 276 | 4 274 | 14 463 | 14 123 |
| **MKG-Y** | Xu et al. 2022 (YAGO) | 15 000 | 28 | 21 310 | 2 665 | 2 663 | 14 244 | 12 305 |
| **DB15K** | Liu et al. 2019 (DBpedia) | 12 842 | 279 | 79 222 | 9 902 | 9 904 | 12 818 | 9 078 |

**Nhận xét tự suy từ bảng** (không phải tác giả viết):
- Chia ≈ 80/10/10 ở cả ba tập.
- Không phải thực thể nào cũng đủ ảnh/text. DB15K chỉ ~71% có text, MKG-Y ~82%. Vì thế trộn mô thức phải chịu được thiếu/nhiễu.
- MKG-Y chỉ 28 quan hệ, mỗi quan hệ dày, cấu trúc đã đủ mạnh (RotatE thuần đạt 36.67 MRR).
- DB15K nhiều quan hệ (279) và nhiều bộ ba nhất, đồ thị dày nhất.

**Định dạng file trong `original/data/`** (copy từ repo MDBGF, cùng benchmark):
- `train/valid/test.txt`: mỗi dòng `head_id  relation_id  tail_id`.
- `entities.txt`, `relations.txt`: danh sách id.
- `entity2id.txt`: tên → id (MKG-W dùng Wikidata QID, MKG-Y dùng tên YAGO).
- Ảnh và text gốc **không** nằm trong folder; các repo MMKGC thường phát hành đặc trưng đã encode.

**Ví dụ input – output** (dùng cho slide):
- Input: `(Gwen Stefani, voice type, ?)` + ảnh Gwen Stefani + text "American singer, songwriter…" + láng giềng trong KG.
- Output: bảng xếp hạng toàn bộ thực thể, top-1 = `Mezzo-soprano` (Figure 4, tr. 7).

---

## 8. Đánh giá

**Cách đo:** với mỗi bộ ba test, che tail (và che head), chấm điểm **toàn bộ thực thể**, xem đáp án thật đứng hạng mấy.

| Metric | Nghĩa |
|---|---|
| **MRR** | trung bình 1/rank. Hạng 1 → 1.0, hạng 2 → 0.5, hạng 10 → 0.1. Đo chất lượng xếp hạng tổng thể. |
| **Hits@1** | % đáp án đúng ở hạng 1. Khó nhất. |
| **Hits@3** | % đáp án đúng trong top 3. |
| **Hits@10** | % đáp án đúng trong top 10. Đo khả năng "không bỏ sót". |

Tất cả tính theo %, càng cao càng tốt. LLM thuần chỉ sinh **10 ứng viên**, không chấm hết mọi thực thể, nên **không tính được MRR** (ký hiệu "—").

### 8.1 Kết quả chính (Table 1, tr. 6, đơn vị %)

| Model | MKG-W MRR | H@10 | H@3 | H@1 | MKG-Y MRR | H@10 | H@3 | H@1 | DB15K MRR | H@10 | H@3 | H@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RotatE (chỉ cấu trúc) | 32.92 | 44.23 | 35.91 | 26.68 | 36.67 | 41.11 | 35.91 | 33.92 | 32.07 | 51.40 | 38.57 | 21.23 |
| RSME | 33.41 | 44.31 | 35.80 | 27.59 | 38.03 | 44.02 | 35.80 | 34.28 | 33.72 | 50.39 | 39.23 | **24.45** |
| MyGO | 33.40 | 44.36 | 35.38 | 27.52 | 30.29 | 42.17 | 35.38 | 23.75 | 32.25 | 48.23 | 35.49 | 24.19 |
| AdaMF-MAT | 32.09 | 45.78 | 35.54 | 24.48 | 37.78 | 46.11 | 35.54 | 32.65 | 32.08 | 52.42 | 40.41 | 19.81 |
| APKGC | 32.67 | 46.20 | 35.83 | 25.24 | 31.48 | 40.68 | 35.83 | 26.17 | 32.91 | 48.91 | 37.15 | 24.39 |
| **NativeE** (baseline mạnh nhất) | 36.79 | 50.28 | 40.47 | 29.36 | 38.34 | 46.24 | 40.47 | 33.49 | 33.59 | 53.77 | 40.67 | 22.11 |
| LLaMA-2-7B (chỉ prompt) | — | 10.20 | 8.73 | 5.62 | — | 2.29 | 2.03 | 1.35 | — | 14.14 | 11.87 | 7.15 |
| LLaVA-1.5-7B (chỉ prompt) | — | 26.42 | 22.29 | 15.02 | — | 3.58 | 2.80 | 1.64 | — | 15.89 | 13.15 | 8.32 |
| **HFR-MKGC** | **38.62** | **51.66** | **42.12** | **31.37** | **39.00** | **47.31** | **42.12** | **34.41** | **35.20** | **56.41** | **43.25** | 23.04 |

(Bảng đầy đủ 14 baseline trong PDF. TransE, DistMult, ComplEx, IKRL, SNAG, DHNS đều thấp hơn.)

**Đọc bảng:**
- HFR-MKGC **đứng đầu 11/12 ô**. Ngoại lệ duy nhất: **Hits@1 trên DB15K** (23.04) thua RSME (24.45), APKGC (24.39), MyGO (24.19). Bài nói "SOTA trên mọi metric", nên nêu trung thực trong seminar.
- MKG-W: hơn NativeE +1.83 MRR, +2.01 H@1.
- MKG-Y cải thiện **nhỏ nhất** (+0.66 MRR): ít quan hệ, cấu trúc đã đủ mạnh, đa mô thức thêm ít giá trị.
- **LLM thuần rất kém** (Hits@10 ≤ 26%): LLM một mình không thay được hàm chấm điểm có cấu trúc. LLaVA (có ảnh) hơn LLaMA (không ảnh) ở mọi tập → ảnh có ích.

### 8.2 Ablation trên MKG-W (Table 3, tr. 7, %)

| Cấu hình | MRR | H@10 | H@3 | H@1 | Bỏ gì |
|---|---|---|---|---|---|
| w/o S | 25.67 | 39.42 | 28.81 | 18.02 | bỏ cấu trúc |
| w/o T | 29.67 | 45.61 | 34.92 | 21.25 | bỏ text |
| w/o V | 31.16 | 47.39 | 36.10 | 22.09 | bỏ ảnh |
| w/o VT | 31.78 | 47.33 | 36.41 | 23.22 | bỏ caption ảnh do LLaVA sinh |
| w/o RHF | 31.98 | 50.25 | 39.14 | 20.80 | thay trộn theo quan hệ bằng **trung bình** |
| w/o MER | 33.52 | 50.61 | 39.70 | 23.42 | bỏ suy luận LLaVA |
| w/o PTI | 34.59 | 50.95 | 40.03 | 25.12 | thay nhiễu text/ảnh bằng **nhiễu ngẫu nhiên** |
| w/o MNO | 34.01 | 50.98 | 40.26 | 24.13 | bỏ adversarial training |
| **Full** | **38.62** | **51.66** | **42.12** | **31.37** | |

**Đọc bảng:**
- **Cấu trúc quan trọng nhất**: bỏ S mất ~13 MRR. Text quan trọng hơn ảnh (bỏ T mất 9, bỏ V mất 7.5).
- **RHF đóng góp lớn nhất** trong 4 module: mất 6.6 MRR, H@1 rớt từ 31.4 xuống 20.8, nhưng H@10 gần như không đổi → trộn theo quan hệ chủ yếu giúp **xếp đúng hạng 1**.
- MER đóng góp ~5 MRR. Hard negative có nghĩa (PTI) và adversarial (MNO) mỗi thứ ~4–4.6 MRR. Caption ảnh (VT) ~6.8 MRR.

### 8.3 Độ nhạy siêu tham số (Figure 3, tr. 7)

- Chiều 512 cần **128 negative** mới đạt 38.62; với 32 hoặc 64 negative chỉ ~33.
- Chiều 256 tốt nhất với 64 negative (36.08). Chiều 128 gần như không phụ thuộc số negative (~36).
- Thiếu negative thì 256D có thể **hơn** 512D.
- MRR và Hits@1 nhạy với siêu tham số, Hits@10 ổn định.

### 8.4 Case study (Figure 4, tr. 7)

Query **(Gwen Stefani, voice type, ?)**, đáp án **Mezzo-soprano**.
- Không có RHF (trọng số ngẫu nhiên): ảnh được trọng số cao nhất, đáp án đúng xếp **hạng 2**, hạng 1 là Double bass.
- Có RHF: ảnh chỉ có ngoại hình, không nói gì về giọng; text nói cô là ca sĩ → tăng trọng số text, giảm trọng số ảnh → đúng **hạng 1**.

---

## 9. Kết luận của tác giả và nhận xét của nhóm

**Tác giả kết luận** (tr. 7): ba đóng góp (trộn phân cấp theo quan hệ, LLaVA fine-tune suy luận theo từng bộ ba, hard negative đa mô thức + adversarial). SOTA trên 3 benchmark. Hướng tương lai: khai thác MLLM chính xác hơn cho biểu diễn và suy luận đa mô thức.

**Hạn chế (nhóm tự đánh giá, dùng cho thảo luận):**
1. **Chi phí lớn**: chạy LLaVA-7B cho *mỗi* ảnh (caption), *mỗi* query (suy luận), *mỗi* text (paraphrase). Bài không báo cáo thời gian huấn luyện/suy luận.
2. **Không có code** → khó tái lập. Nhiều baseline do tác giả tự tái hiện.
3. **Cải thiện không đồng đều**: MKG-Y chỉ +0.66 MRR; thua Hits@1 trên DB15K nhưng vẫn tuyên bố SOTA mọi metric.
4. **Ablation chỉ trên MKG-W.**
5. LLaVA chỉ sinh **1 đáp án**; nếu ảo giác tên không tồn tại, cổng phải học bỏ qua. Bài không phân tích tỉ lệ LLaVA đoán đúng hay cổng hoạt động thế nào.
6. **Rò rỉ tri thức tiềm ẩn**: LLaVA/BERT pre-trained trên web có thể đã "biết" sự thật trong Wikidata/DBpedia. Bài không thảo luận.
7. Chi tiết prompt, số epoch LoRA, cách chọn láng giềng không được mô tả.
8. Ký hiệu đôi chỗ chưa nhất quán (H̃_joint chưa định nghĩa).

---

## 10. Q&A dự kiến

| Câu hỏi | Trả lời |
|---|---|
| Tại sao dùng RotatE mà không phải scoring khác? | RotatE mạnh với quan hệ đối xứng, nghịch đảo, hợp thành; biểu diễn phức thuận tiện cho cổng tách thực/ảo. |
| Cổng g tính từ T_MLLM chứ không từ T_joint, vì sao? | Để cổng học "độ tin cậy" của chính đáp án LLaVA. |
| Hard negative đa mô thức khác gì negative thay id? | Thay id tạo bộ ba sai về cấu trúc; đa mô thức giữ nguyên thực thể nhưng bóp méo đặc trưng, ép học biểu diễn bền hơn. |
| Mô hình có ngưỡng đúng/sai không? | Không. Link prediction chỉ xếp hạng. Muốn dùng thật phải tự thêm ngưỡng trên validation, thường theo từng quan hệ vì thang điểm RotatE không chuẩn hoá. |
| Thêm được thực thể mới không? | Không, mô hình **transductive**: mỗi thực thể là vector tra bảng học lúc train. Thực thể mới phải train lại. Bài toán đó là inductive KGC (CATS trong related work). |
| Vì sao LLM thuần kém? | Chỉ sinh 10 ứng viên, không chấm hết tập thực thể; không biết ranh giới tập thực thể; ảo giác. |
| Caption ảnh có thật sự cần? | Có, bỏ caption (w/o VT) mất ~6.8 MRR. |

---

## 11. Bản đồ trang PDF

| Nội dung | Trang |
|---|---|
| Abstract, Introduction, 2 thách thức | 1 |
| Figure 1 (3 paradigm), Contributions, Related Work | 2 |
| Figure 2 (kiến trúc), Task Definition, Encoding, Tầng 1, đầu Tầng 2 | 3 |
| Eq. 2–10: Tầng 2, Instruction FT, Reasoning, Gated Fusion, RotatE score | 4 |
| Eq. 11–14: Perturbation, Augmentation, Adversarial loss; Datasets, Metrics, Baselines | 5 |
| Table 1 (kết quả), Table 2 (dataset), Main Results, mô tả Ablation | 6 |
| Table 3 (ablation), Figure 3 (siêu tham số), Figure 4 (case study), Conclusion | 7 |
| References | 8–9 |
