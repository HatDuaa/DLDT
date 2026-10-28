"""Chuẩn bị dữ liệu cho mã MDBGF của tác giả: giải nén token ảnh và tạo tokens/textual.pth.

Repo tác giả không kèm tokens/textual.pth. Token văn bản trong tokens/<data>-textual.json là id
của bert-base-uncased (từ vựng 30522), nên ma trận nhúng lấy từ chính mô hình đó, giống
save_token_embeddings.py của tác giả.
"""
import argparse
import os
import zipfile

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="thư mục mã nguồn MDBGF của tác giả")
    p.add_argument("--data", nargs="+", default=["MKG-W"])
    p.add_argument("--bert", default="bert-base-uncased")
    args = p.parse_args()
    tokens = os.path.join(args.src, "tokens")

    for data in args.data:
        target = os.path.join(tokens, f"{data}-visual.json")
        if os.path.exists(target):
            print(f"Đã có {target}")
            continue
        with zipfile.ZipFile(target + ".zip") as z:
            z.extract(f"{data}-visual.json", tokens)
        print(f"Đã giải nén {target}")

    textual = os.path.join(tokens, "textual.pth")
    if os.path.exists(textual):
        print(f"Đã có {textual}")
    else:
        from transformers import AutoModel

        weight = AutoModel.from_pretrained(args.bert).get_input_embeddings().weight
        torch.save(weight.detach().clone(), textual)
        print(f"Đã tạo {textual} {tuple(weight.shape)}")


if __name__ == "__main__":
    main()
