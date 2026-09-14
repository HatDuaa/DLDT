"""Build seminar-slides.html from the template: embed figures as base64 and inject speaker notes."""
import base64
import io
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
FIG_DIR = HERE.parent / "paper-vi" / "figures"
SRC_SELF = "https://claude.ai/code/artifact/79ab1978-2487-4ea1-9077-c9996e7cbef2"

NOTES = [
    # 1
    "Chào thầy và các bạn. Nhóm trình bày bài HFR-MKGC, AAAI 2026, về hoàn thiện đồ thị tri thức đa phương thức. Tôi (Lộc) nói phần bài toán và kiến trúc, Đạt nói phần thực nghiệm và nhận xét.",
    # 2
    "Bài toán là link prediction trên KG: biết head và quan hệ, tìm tail. Không có node mới, chỉ thiếu cạnh. Khác KGC thường ở chỗ mỗi node còn có ảnh và văn bản. Câu hỏi của bài: dùng ảnh và văn bản thế nào để không bị nhiễu.",
    # 3
    "Ví dụ xuyên suốt: Gwen Stefani, quan hệ voice type. Input là ba loại thông tin của Gwen cộng tên quan hệ. Output là bảng xếp hạng toàn bộ 15 nghìn node. Điểm mấu chốt: ảnh chỉ có ngoại hình, không nói gì về giọng; văn bản có chữ singer. Mô hình phải biết nghe văn bản nhiều hơn với câu hỏi này.",
    # 4
    "Hai điểm yếu tác giả chỉ ra. Một: các phương pháp trước trộn ba mô thức với trọng số cố định, không tuỳ quan hệ. Hai: các phương pháp dùng LLM chỉ dùng text và cấu trúc, chưa dùng ảnh và không thích nghi. Hình 1 của bài so sánh ba mô hình: truyền thống, LLM-based, và HFR-MKGC gộp cả hai.",
    # 5
    "Giải pháp là ba mô-đun đặt lên nền RotatE. RHF trộn mô thức theo quan hệ. MER cho LLaVA đoán đáp án rồi trộn vào qua cổng. MNO tạo mẫu sai khó và train đối kháng. Lưu ý: không có GNN, node là hàng tra bảng, cấu trúc đi vào qua loss.",
    # 6
    "Hình 2 của bài. Trái: mã hoá ba mô thức và tầng 1. Phải trên: tầng 2 và MER, thấy rõ prompt vào MLLM, ra reference answers, gặp cổng G rồi mới vào score function. Phải dưới: MNO sinh negative. Ta đi lần lượt từng khối.",
    # 7
    "Trước hết chốt ký hiệu. Mỗi node có ba vector: S tra bảng học từ đầu, I từ CLIP, T từ BERT. Hai encoder đóng băng, chỉ học lớp chiếu để về cùng chiều d. Quan hệ là một vector duy nhất, không có mô thức. Cái được lưu là ba vector; cái được tính là tổ hợp của chúng, tỉ lệ đổi theo quan hệ.",
    # 8
    "Tầng 1 chỉ xử lý ảnh. Ảnh được mã hoá hai cách: CLIP ra vector thô, LLaVA viết caption rồi BERT ra vector ngữ nghĩa. Cổng là sigmoid của tích vô hướng: hai vector đồng ý thì tin caption, không đồng ý thì giữ CLIP. Không có tham số học ở cổng; cái học là hai lớp chiếu, và chúng học gián tiếp qua loss KGC.",
    # 9
    "Tầng 2 là trái tim của bài. Điểm thô của mỗi mô thức gồm hai phần: hợp với quan hệ bao nhiêu, và nhất quán với hai mô thức kia bao nhiêu. Softmax ra ba tỉ lệ, pha lại. Với voice type, text được 55%, ảnh 10%. Đổi quan hệ thì số hạng đầu đổi, tỉ lệ đảo. Đây là attention nhưng không có ma trận W: query là r trần, key là vector mô thức trần.",
    # 10
    "MER. Prompt gồm text, láng giềng, ảnh và quan hệ, vào LLaVA đã fine-tune LoRA, ra một cái tên. Tên đó BERT hoá thành T_MLLM. Cổng g là MLP hai lớp, chỉ nhìn T_MLLM, ra tỉ lệ tin từng chiều. Rồi mỗi ứng viên được kéo về phía T_MLLM theo g. Hạn chế: g không nhìn câu hỏi, chỉ nhìn đáp án, nên LLaVA đoán một tên có thật mà sai thì cổng khó nhận ra.",
    # 11
    "Hàm điểm là RotatE. Quan hệ đổi thành góc, xoay head trên từng mặt phẳng phức, đo khoảng cách tới ứng viên. Gần thì điểm cao. Chọn RotatE vì một phép xoay biểu diễn được đối xứng, nghịch đảo, hợp thành. Cài đặt hoàn toàn bằng số thực.",
    # 12
    "MNO. Mẫu sai thông thường thay id, quá dễ. MNO giữ node, bóp méo đặc trưng: LLaVA viết lại text, ma trận xoay cộng nhiễu cho ảnh, cường độ tăng theo epoch. Đây là GAN nhỏ: generator là bộ bóp méo, discriminator là hàm điểm, có gradient penalty. Nhóm lưu ý: paraphrase cùng nghĩa thì mẫu giả gần như vẫn đúng; bài không kiểm chứng.",
    # 13
    "Gom lại luồng dữ liệu cho một truy vấn. Trên: ba mô thức của head qua tầng 1, tầng 2 ra H_joint. Giữa: prompt qua LLaVA, BERT, ra T_MLLM và cổng. Mỗi ứng viên qua cùng RHF ra T_joint, qua cổng ra T tilde, rồi RotatE chấm 15 nghìn điểm. Lúc train thêm mẫu sai và loss. (Chuyển sang Đạt.)",
    # 14
    "Ba benchmark chuẩn. MKG-W từ Wikidata, MKG-Y từ YAGO, DB15K từ DBpedia. Không phải node nào cũng đủ ảnh và text, DB15K thiếu text tới 29%. Độ đo là MRR và Hits@K, hỏi cả hai chiều, filtered. Mọi baseline được chạy lại với cùng CLIP/BERT.",
    # 15
    "Kết quả chính. HFR-MKGC hơn NativE, baseline tốt nhất, khoảng 1.6 đến 1.8 MRR trên MKG-W và DB15K, nhưng chỉ 0.66 trên MKG-Y vì tập này ít quan hệ, cấu trúc đã đủ. Bài nói SOTA mọi metric, thực ra Hits@1 trên DB15K thua ba baseline. LLaVA thuần rất kém, chứng tỏ cần hàm điểm có cấu trúc.",
    # 16
    "Ablation trên MKG-W. Bỏ RHF mất nhiều nhất trong bốn mô-đun, và đặc biệt Hits@1 rớt mạnh còn Hits@10 giữ nguyên: trộn theo quan hệ giúp xếp đúng hạng 1. Bỏ cấu trúc mất 13 điểm, vẫn là mô thức quan trọng nhất. Caption do LLaVA sinh có ích thật.",
    # 17
    "Case study khép lại ví dụ Gwen Stefani. Không RHF thì ảnh chiếm trọng số cao nhất, đáp án đứng hạng 2. Có RHF, text lên ảnh xuống, hạng 1.",
    # 18
    "Nhận xét của nhóm. Chi phí LLaVA lớn và không được báo cáo. Không có code. Cải thiện không đều. Cổng G không nhìn câu hỏi. Mẫu giả chưa chắc sai. Căn chỉnh CLIP với BERT chỉ gián tiếp. Và nhiều siêu tham số không được nêu.",
    # 19
    "Kết luận một câu: HFR-MKGC là RotatE cộng trộn mô thức có dẫn hướng bởi quan hệ, cộng LLaVA làm cố vấn có kiểm soát, cộng GAN nhỏ tạo mẫu sai. Đóng góp thật nằm ở tầng 2 của RHF. Cảm ơn thầy và các bạn.",
    # 20
    "Tài liệu tham khảo.",
]


def embed(name: str) -> str:
    im = Image.open(FIG_DIR / f"{name}.png").convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, int(im.height * 1600 / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    html = (HERE / "seminar-slides-template.html").read_text(encoding="utf-8")
    for i in (1, 2, 4):
        html = html.replace("{{FIG%d}}" % i, embed(f"fig{i}"))
    html = html.replace("{{SRC_SELF}}", SRC_SELF)
    html = html.replace("{{NOTES_JSON}}", json.dumps(NOTES, ensure_ascii=False))
    assert "{{" not in html, "unfilled placeholder"
    out = HERE / "seminar-slides.html"
    out.write_text(html, encoding="utf-8")
    print(out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
