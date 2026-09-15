# Script nói — Seminar HFR-MKGC

Tổng thời gian mục tiêu: **14 phút** (trần 15). Lộc nói slide 1–12 (~8 phút), Đạt nói slide 13–19 (~6 phút).
Chữ in nghiêng trong ngoặc là hành động (chỉ hình, chuyển slide). Không đọc nguyên văn chữ trên slide; slide chỉ là điểm neo.

Nguyên tắc khi nói:

- Mỗi slide mở đầu bằng **một câu nói ý chính**, rồi mới đi vào hình.
- Luôn kéo về ví dụ Gwen Stefani / voice type để người nghe có một sợi chỉ xuyên suốt.
- Không đọc công thức từng ký hiệu. Nói "phần này là gì, làm gì", rồi chỉ vào màu tương ứng.

---

## Slide 1 — Tiêu đề (0:20) — Lộc

Chào thầy và các bạn. Nhóm em gồm Lộc và Đạt, trình bày bài **HFR-MKGC**, đăng tại AAAI 2026, về bài toán hoàn thiện đồ thị tri thức đa phương thức. Em sẽ nói phần bài toán và kiến trúc, bạn Đạt nói phần thực nghiệm và nhận xét.

*(chuyển slide)*

## Slide 2 — Bài toán và ví dụ (1:10)

Bài toán là **link prediction trên knowledge graph**. *(chỉ hình)* Ta có thực thể Gwen Stefani, gồm ảnh, đoạn mô tả và vị trí trong đồ thị. Đã biết cô ấy là thành viên của No Doubt và làm trong lĩnh vực âm nhạc. Câu hỏi là quan hệ voice type nối tới node nào. *(chỉ khung ứng viên)* Đáp án không được sinh ra mà phải chọn trong **toàn bộ** 15 nghìn thực thể của tập dữ liệu: Mezzo-soprano, Double bass, Soprano Home Movies, và mọi node khác đều là ứng viên. Soprano Home Movies là một tập phim truyền hình, lọt vào top chỉ vì tên có chữ Soprano, đó là kiểu nhiễu do trùng từ. Mô hình cho điểm từng ứng viên rồi xếp hạng, đáp án đúng nằm càng cao càng tốt.

Điểm mấu chốt nằm ở ô xanh: ảnh chân dung **không chứa thông tin gì về giọng hát**, còn văn bản có từ "singer". Một mô hình tốt phải biết với câu hỏi này thì ưu tiên văn bản hơn ảnh. Toàn bộ bài báo xoay quanh ý đó. (Nếu bị hỏi: trong MKG-W, Gwen chỉ có một cạnh train là field of work → music, cạnh voice type nằm trong tập test.)

Điểm khác với KGC thường: ở đây mỗi node còn có **ảnh** và **đoạn mô tả văn bản**, gọi là multi-modal. Câu hỏi của bài báo là: dùng hai nguồn đó thế nào để giúp chứ không gây nhiễu.

## Slide 3 — Hai điểm yếu (0:45)

Tác giả chỉ ra hai điểm yếu của các phương pháp trước.

Một: *(chỉ thẻ trên)* trộn ba mô thức với **trọng số cố định**, không phụ thuộc quan hệ đang hỏi. Ảnh không liên quan tới voice type nhưng vẫn được trọng số như văn bản, thành nhiễu.

Hai: *(chỉ thẻ dưới)* các phương pháp dùng LLM thì chỉ đưa văn bản và cấu trúc vào prompt, chưa dùng ảnh, và không thích nghi theo từng bộ ba.

*(chỉ hình phải)* Hình 1 của bài so ba mô hình: (a) truyền thống encoder rồi fusion rồi score; (b) dựa trên LLM, biến đồ thị thành prompt; (c) HFR-MKGC gộp cả hai, và thêm hai thứ: fusion có dẫn hướng bởi quan hệ, và cổng G nối đáp án LLM vào hàm điểm.

## Slide 4 — Ba mô-đun (0:40)

Giải pháp là ba mô-đun đặt lên nền RotatE.

*(chỉ từng thẻ)* **RHF** trộn mô thức theo quan hệ, hai tầng. **MER** cho LLaVA đoán thẳng tên đáp án rồi đưa vào hàm điểm qua một cổng học được. **MNO** tạo mẫu sai khó bằng cách bóp méo ảnh và văn bản, huấn luyện đối kháng.

Nền là RotatE làm hàm điểm, CLIP và BERT đóng băng làm encoder, LLaVA-1.5-7B làm MLLM.

Một điều cần nói rõ để tránh nhầm với các bài trong môn: bài này **không dùng GNN**. Mỗi node là một hàng trong bảng embedding, cấu trúc đồ thị đi vào mô hình qua hàm loss chứ không qua truyền tin giữa các node.

## Slide 5 — Kiến trúc tổng thể (0:35)

Đây là hình 2 của bài. *(chỉ trái)* Bên trái là mã hoá ba mô thức và tầng 1 của RHF. *(chỉ phải trên)* Phải trên là tầng 2 và MER: thấy prompt đi vào MLLM, ra "reference answers", gặp cổng G rồi mới tới score function. *(chỉ phải dưới)* Phải dưới là MNO sinh mẫu sai.

Hình này khá dày, nên em sẽ đi từng khối một, bắt đầu từ chuyện mỗi node được biểu diễn thế nào.

## Slide 6 — Một node = ba vector (0:45)

Chốt ký hiệu trước để phần sau đỡ rối.

*(chỉ hình)* Mỗi node có **ba vector**. S là cấu trúc: một hàng trong bảng, khởi tạo ngẫu nhiên, học từ đầu. I là ảnh: qua CLIP, cộng thêm caption do LLaVA sinh rồi qua BERT. T là văn bản: qua BERT. Hai encoder CLIP và BERT **đóng băng**, chỉ học hai lớp chiếu P để đưa mọi thứ về cùng chiều d.

*(chỉ phải)* Quan hệ r là **một vector duy nhất**, cũng học từ đầu, không có ảnh hay văn bản gì.

Điều quan trọng: cái được **lưu** là ba vector cố định. Cái được **tính** là tổ hợp của ba vector đó với tỉ lệ đổi theo quan hệ. Nên node có nhiều biểu diễn nhưng số tham số không tăng.

## Slide 7 — RHF tầng 1 (0:50)

Tầng 1 chỉ xử lý **ảnh**. Lý do: ảnh được mã hoá hai cách. *(chỉ hình dưới)* Nhánh trên CLIP ra vector thị giác thô. Nhánh dưới LLaVA viết caption, ví dụ "a woman singing on stage", rồi BERT ra vector ngữ nghĩa. Hai vector nói về cùng một tấm ảnh, cần gộp thành một.

*(chỉ công thức)* Cách gộp là một cổng: tích vô hướng đo hai vector có tương đồng không, sigmoid biến thành tỉ lệ. *(chỉ màu vàng)* Tương đồng cao thì tin caption nhiều hơn, vì caption đã ở dạng ngôn ngữ, gần với thế giới của KG. Tương đồng thấp thì caption có thể sai, giữ CLIP.

Lưu ý cổng này **không có tham số**. Cái học được là hai lớp chiếu đưa CLIP 512 chiều và BERT 768 chiều về cùng không gian, và chúng học gián tiếp qua loss cuối cùng.

## Slide 8 — RHF tầng 2 (1:00)

Tầng 2 là phần quan trọng nhất của bài, cũng là chỗ "relation-guided" trong tên.

*(chỉ công thức 2)* Với mỗi mô thức m, tính một điểm thô gồm hai phần. *(chỉ màu xanh)* Phần một: tích vô hướng của vector mô thức với vector quan hệ, tức mô thức này liên quan tới quan hệ bao nhiêu. Đây là chỗ làm trọng số **đổi theo r**. *(chỉ màu đỏ nhạt)* Phần hai: trung bình tương đồng với hai mô thức kia, để hạ những mô thức bị thiếu hoặc bịa.

*(chỉ công thức 3, 4)* Softmax ra ba tỉ lệ cộng bằng 1, rồi pha ba vector theo tỉ lệ đó thành H_joint.

*(chỉ biểu đồ)* Với Gwen và voice type, văn bản được khoảng 55%, ảnh chỉ 10%. Đổi sang quan hệ khác, số hạng đầu đổi, tỉ lệ đảo lại. Cùng node, khác quan hệ, khác vector.

Về bản chất đây là attention, nhưng không có ma trận W: query là chính r, key là chính vector mô thức.

## Slide 9 — MER (1:00)

Mô-đun MER cho LLaVA tham gia.

*(chỉ bước 1)* Ghép prompt: phần cố định mô tả nhiệm vụ, cộng văn bản của Gwen, tên các láng giềng, ảnh, và chữ "voice type". Đưa vào LLaVA đã fine-tune bằng LoRA. Ra **một cái tên**, ví dụ "Mezzo-soprano".

*(bước 2)* Tên đó qua BERT thành vector T_MLLM. Chỉ tính một lần cho mỗi câu hỏi.

*(bước 3)* Cổng g là một MLP hai lớp, vào 2d, ẩn d, ra 2d. Đây là chỗ **có tham số học**, khác hai cổng ở RHF. Nó quyết định tin đáp án LLaVA bao nhiêu ở từng chiều.

*(bước 4, chỉ hình)* Với mỗi ứng viên tail, pha vector của ứng viên với T_MLLM theo tỉ lệ g. Tức mọi ứng viên bị kéo về phía đáp án LLaVA một chút; ứng viên nào vốn đã gần đáp án thì càng gần.

*(chỉ ô đỏ)* Hạn chế nhóm thấy: g chỉ nhìn đáp án, không nhìn câu hỏi. Nếu LLaVA đoán một tên có thật nhưng sai, cổng khó nhận ra.

## Slide 10 — RotatE (0:45)

Hàm điểm là RotatE, chắc các bạn đã học. Em chỉ nhắc cách nó ghép vào đây.

*(chỉ công thức trái)* Vector quan hệ được chia cho một hằng số γ rồi lấy cos, sin, thành d góc xoay, mỗi chiều phức một góc. *(chỉ hình)* Trên mỗi mặt phẳng, xoay H_joint theo góc đó ra điểm neo, rồi đo khoảng cách tới T̃ của ứng viên. Gần thì điểm cao.

Xoay chỉ làm một lần, đo thì 15 nghìn lần, một lần cho mỗi ứng viên.

Chọn RotatE vì một phép xoay biểu diễn được cả ba kiểu quan hệ: đối xứng, nghịch đảo, hợp thành. Và cài đặt hoàn toàn bằng số thực, không cần kiểu số phức.

## Slide 11 — MNO (0:55)

Mô-đun cuối là về mẫu sai lúc train.

Mẫu sai thông thường là thay tail bằng node ngẫu nhiên, quá dễ phân biệt, mô hình học được ít. MNO giữ nguyên node nhưng **bóp méo đặc trưng**. *(chỉ thẻ vàng)* Văn bản: LLaVA viết lại cùng nghĩa khác lời. *(chỉ thẻ tím)* Ảnh: nhân vector CLIP với ma trận xoay R rồi cộng nhiễu Gauss; λ tăng dần theo epoch nên mẫu giả dễ trước, khó sau.

*(chỉ hình phải)* Đây là một GAN nhỏ. Generator là bộ bóp méo, discriminator chính là hàm điểm. Loss L_g phạt generator khi mẫu giả bị nhận ra dễ, tức điểm dưới ngưỡng c. Có gradient penalty kiểu WGAN-GP để ổn định. Khung đối kháng này kế thừa từ AdaMF-MAT và MACO; phần mới của bài là cách sinh mẫu giả bằng LLM và ma trận xoay.

*(chỉ ô đỏ)* Nhóm lưu ý: paraphrase "cùng nghĩa" và ảnh xoay nhẹ về bản chất vẫn là node gốc. Bài coi chúng là sai nhưng không kiểm chứng.

## Slide 12 — Luồng dữ liệu (0:50)

Gom lại toàn bộ cho một truy vấn.

*(chỉ từ trái sang)* Ba mô thức của head: ảnh qua CLIP và qua LLaVA-BERT, gộp ở tầng 1; văn bản qua BERT; id tra bảng. Cả ba vào tầng 2 cùng quan hệ r, ra H_joint.

*(chỉ hàng dưới)* Song song, prompt vào LLaVA, ra tên, BERT hoá thành T_MLLM, tính cổng g.

*(chỉ khối ứng viên)* Mỗi ứng viên trong 15 nghìn node đi qua cùng RHF ra T_joint, qua cổng G ra T̃.

*(chỉ khối đen)* RotatE xoay H_joint theo r, đo khoảng cách tới từng T̃, xếp hạng.

Lúc train, chạy y hệt, chỉ thêm mẫu sai từ MNO và tính loss, rồi cập nhật bảng E, bảng R, lớp chiếu, cổng và LoRA cùng lúc.

Phần thực nghiệm xin mời bạn Đạt.

---

## Slide 13 — Dữ liệu và độ đo (0:50) — Đạt

Cảm ơn Lộc. Bài chạy trên ba benchmark chuẩn của MMKGC. *(chỉ bảng)* MKG-W từ Wikidata, 15 nghìn thực thể, 169 quan hệ. MKG-Y từ YAGO, cùng số thực thể nhưng chỉ 28 quan hệ, nên cấu trúc đã rất mạnh. DB15K từ DBpedia, nhiều bộ ba nhất.

Hai cột cuối nhóm tự tính: không phải node nào cũng đủ ảnh và văn bản, DB15K thiếu văn bản tới gần 30%. Đó là lý do fusion phải chịu được mô thức thiếu.

Độ đo là MRR và Hits@K. Mỗi bộ ba test hỏi cả hai chiều, lọc các đáp án đúng khác. Tác giả chạy lại 14 baseline với cùng CLIP và BERT để so cho công bằng.

## Slide 14 — Kết quả chính (1:00)

*(chỉ biểu đồ)* Ba cụm cột là ba tập, mỗi cụm: RotatE thuần, NativE là baseline mạnh nhất, và HFR-MKGC.

HFR-MKGC hơn NativE 1.83 MRR trên MKG-W, 1.61 trên DB15K, nhưng chỉ 0.66 trên MKG-Y. Lý do như Lộc nói: MKG-Y ít quan hệ, cấu trúc đã đủ, đa mô thức thêm ít.

Tổng cộng đứng đầu 11 trên 12 ô. *(chỉ ô đỏ)* Ngoại lệ là Hits@1 trên DB15K, thua RSME, APKGC và MyGO. Bài viết "SOTA trên mọi metric", nhóm thấy cần nói chính xác là 11/12.

Dòng cuối: LLaVA dùng một mình, không có hàm điểm, Hits@10 chỉ 26% trên MKG-W và 3.6% trên MKG-Y. Tức LLM không thay được hàm điểm có cấu trúc, chỉ bổ trợ.

## Slide 15 — Ablation (0:55)

Ablation chỉ làm trên MKG-W. *(chỉ thanh)* Thanh đỏ là mô hình đầy đủ, 38.62.

Bốn thanh xám là bỏ từng mô-đun. Bỏ RHF mất nhiều nhất, xuống 31.98. Và đáng chú ý: Hits@1 rớt từ 31.4 xuống 20.8 nhưng Hits@10 gần như giữ nguyên. Nghĩa là trộn theo quan hệ không giúp tìm thêm ứng viên, mà giúp **xếp đúng ứng viên lên hạng 1**.

Bốn thanh màu là bỏ từng mô thức. Bỏ cấu trúc mất 13 điểm, vẫn là mô thức quan trọng nhất. Văn bản hơn ảnh. Bỏ riêng caption do LLaVA sinh mất gần 7 điểm, nên tầng 1 có ích thật.

## Slide 16 — Case study (0:35)

Quay lại ví dụ Gwen Stefani, đây là hình 4 của bài.

*(chỉ trái dưới)* Không có RHF, trọng số ngẫu nhiên: ảnh có trọng số cao nhất dù chỉ có ngoại hình, Mezzo-soprano đứng hạng 2. *(chỉ phải dưới)* Có RHF: trọng số văn bản tăng, ảnh giảm, Mezzo-soprano lên hạng 1. Đúng như ý ở slide 2.

## Slide 17 — Nhận xét và hạn chế (1:00)

Nhóm có sáu nhận xét.

*(chỉ từng thẻ)* Chi phí: LLaVA-7B chạy cho mỗi ảnh, mỗi truy vấn, mỗi đoạn văn bản, bài không báo cáo thời gian. Tái lập: không công khai code, nhiều baseline tác giả tự chạy lại. Cải thiện không đều như đã nói. Cổng G không dùng ngữ cảnh câu hỏi. Mẫu giả trong MNO chưa được kiểm chứng là sai thật; hướng này ngược với MDBGF, bài dùng contrastive để kéo các mô thức của cùng node lại gần. Và việc so CLIP với BERT bằng tích vô hướng chỉ có nghĩa nhờ lớp chiếu học gián tiếp, không có loss căn chỉnh riêng.

Ngoài ra nhiều siêu tham số không được nêu: prompt, epoch LoRA, cách chọn láng giềng, γ, c, τ.

## Slide 18 — Kết luận (0:35)

Kết luận. Ý chính của bài: trọng số trộn mô thức phải phụ thuộc quan hệ đang xét. Phương pháp: giữ RotatE, thay đầu vào bằng vector đã trộn theo r, thêm dự đoán MLLM qua cổng, train với mẫu sai khó. Bằng chứng: +1.8 MRR trên MKG-W, RHF đóng góp nhiều nhất, cấu trúc vẫn quan trọng nhất.

*(chỉ ô xanh)* Gọn lại: HFR-MKGC là RotatE cộng trộn mô thức có dẫn hướng bởi quan hệ, cộng dự đoán MLLM có kiểm soát, cộng huấn luyện đối kháng.

Cảm ơn thầy và các bạn. Nhóm sẵn sàng nhận câu hỏi.

## Slide 19 — Tài liệu tham khảo

*(để trên màn hình khi Q&A, không đọc)*

---

## Q&A — câu trả lời ngắn

**Đây có phải GNN không?** Không. Là knowledge graph embedding kiểu tra bảng, không có message passing. Cấu trúc đi vào qua loss: mỗi cạnh trong train là một ràng buộc "h xoay theo r phải gần t".

**Vậy có bất biến hoán vị không?** Câu hỏi này chỉ phát sinh khi có phép gom hàng xóm. Ở đây không có phép gom, vector node là hàng cố định, nên không đặt ra. Đánh số lại node chỉ đổi vị trí hàng trong bảng, kết quả không đổi.

**Thêm thực thể mới được không?** Không, transductive. Node mới không có hàng trong bảng, phải train lại.

**Hai node không có quan hệ thì sao?** Bài không đặt ngưỡng. Đánh giá là xếp hạng, không quyết định có/không. Muốn dùng thật phải thêm ngưỡng trên validation.

**Vì sao dùng RotatE?** Một phép xoay biểu diễn được đối xứng, nghịch đảo, hợp thành; và biểu diễn phức tách thực/ảo tiện cho cổng G. Tác giả không đổi hàm điểm, chỉ đổi thứ đưa vào.

**Số phức lưu thế nào?** Mỗi chiều phức là hai số thực, vector d chiều phức lưu bằng 2d số. Phép xoay tính bằng cos, sin.

**Cổng ở tầng 1 và tầng 2 có học không?** Không có tham số. Chỉ tích vô hướng, sigmoid, softmax. Tham số học nằm ở lớp chiếu P_I, P_T, bảng E, bảng R. Cổng G ở MER thì có tham số, là MLP hai lớp.

**Vector CLIP và BERT khác không gian, sao so được?** Không so trực tiếp. Hai lớp chiếu học được đưa về cùng không gian, và chúng học end-to-end qua loss KGC. Bài không có loss căn chỉnh riêng, đó là điểm yếu.

**LLaVA có được train không?** Có LoRA cho nhiệm vụ đoán tail. Caption và paraphrase thì chỉ chạy suy luận. CLIP, BERT đóng băng hoàn toàn. Khúc nối CLIP với LLM là có sẵn trong LLaVA, nhóm tác giả không nối.

**Mẫu giả trong MNO có chắc sai không?** Không chắc, bài không kiểm chứng. Với λ nhỏ mẫu giả gần trùng node gốc. Cách hiểu hợp lý: loss chỉ ép điểm bản gốc cao hơn bản giả một chút, làm ranh giới quanh mỗi node sắc hơn.

**Vì sao LLM thuần kém?** Chỉ sinh 10 ứng viên, không chấm toàn bộ tập, không biết ranh giới tập thực thể, hay bịa tên.

**Bài có chuẩn hoá vector trước tích vô hướng không?** Không đề cập, không có code để kiểm tra. MDBGF thì dùng cosine có chuẩn hoá.

**Khác MDBGF chỗ nào?** MDBGF căn chỉnh mô thức tường minh bằng contrastive và dùng dropout mô thức; HFR-MKGC căn chỉnh gián tiếp, dùng LLM và mẫu sai đối kháng. Hai giả định ngược nhau về việc biến thể của một node nên gần hay xa.
