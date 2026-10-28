"""Huấn luyện lại MDBGF: ghi WandB, chạy tiếp từ checkpoint, chọn kết quả theo tập kiểm định.

Dựa trên train_mygo_fgc.py trong mã tác giả (github.com/ferryman-ship/MDBGF, commit c512d79).
Mã tác giả được import từ thư mục --src và không bị sửa. Những gì khác so với script gốc:
  * --md chọn cách rụng mô thức: "code" (hằng số alpha, đúng mã gốc), "paper" (lịch tuyến tính
    theo công thức (1) của bài), "none" (không rụng, dùng cho thí nghiệm bỏ MD).
  * --no_bgf bỏ hai cổng BGF (biến thể của nhóm; bài không nói rõ cơ chế thay thế).
  * Đánh giá theo lô thay vì từng bộ ba, cùng quy tắc xếp hạng có lọc như utils.calculate_rank.
  * Ghi cả hai cách chọn kết quả: theo MRR tập kiểm định và theo MRR tập kiểm thử (như mã gốc).
  * Lưu checkpoint định kỳ và tự chạy tiếp nếu --out đã có last.ckpt (phiên Colab bị ngắt).
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="thư mục mã nguồn MDBGF của tác giả")
    p.add_argument("--out", required=True, help="thư mục lưu checkpoint và log của lần chạy")
    p.add_argument("--name", default=None, help="tên run trên WandB (mặc định: tên thư mục --out)")
    p.add_argument("--data", default="MKG-W")
    # Siêu tham số mặc định theo run.sh của tác giả cho MKG-W.
    p.add_argument("--num_epoch", default=5000, type=int)
    p.add_argument("--valid_epoch", default=50, type=int)
    p.add_argument("--lr", default=6e-4, type=float)
    p.add_argument("--dim", default=256, type=int)
    p.add_argument("--hidden_dim", default=1024, type=int)
    p.add_argument("--num_head", default=4, type=int)
    p.add_argument("--num_layer_enc_ent", default=1, type=int)
    p.add_argument("--num_layer_enc_rel", default=1, type=int)
    p.add_argument("--num_layer_dec", default=2, type=int)
    p.add_argument("--dropout", default=0.01, type=float)
    p.add_argument("--emb_dropout", default=0.9, type=float)
    p.add_argument("--vis_dropout", default=0.4, type=float)
    p.add_argument("--txt_dropout", default=0.1, type=float)
    p.add_argument("--smoothing", default=0.0, type=float)
    p.add_argument("--batch_size", default=2048, type=int)
    p.add_argument("--decay", default=0.0, type=float)
    p.add_argument("--max_img_num", default=3, type=int)
    p.add_argument("--step_size", default=50, type=int)
    p.add_argument("--max_vis_token", default=16, type=int)
    p.add_argument("--max_txt_token", default=24, type=int)
    p.add_argument("--score_function", default="tucker")
    # Ba mô-đun của MDBGF.
    p.add_argument("--md", default="code", choices=["code", "paper", "none"])
    p.add_argument("--alpha", default=0.3, type=float, help="xác suất rụng tối đa")
    p.add_argument("--no_bgf", action="store_true")
    p.add_argument("--mu", default=0.01, type=float, help="trọng số lambda của L_AHNS; 0 là bỏ AHNS")
    p.add_argument("--beta", default=0.8, type=float, help="tỉ lệ mẫu âm khó tối đa")
    p.add_argument("--num_neg", default=20, type=int)
    # Vận hành.
    p.add_argument("--seed", default=2024, type=int)
    p.add_argument("--eval_batch", default=512, type=int)
    p.add_argument("--max_hours", default=0.0, type=float,
                   help="dừng và lưu checkpoint sau ngần này giờ (0 là không giới hạn)")
    p.add_argument("--wandb_project", default="mdbgf-reproduce")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def hybrid_negative_ce_loss(score, label, num_neg=20, hard_ratio=0.5, smoothing=0.1):
    """AHNS: chép nguyên từ train_mygo_fgc.py của tác giả."""
    batch_size, num_entities = score.shape
    device = score.device
    num_hard = int(num_neg * hard_ratio)
    num_rand = num_neg - num_hard

    pos_score = score[torch.arange(batch_size), label]
    masked_score = score.clone()
    masked_score[torch.arange(batch_size), label] = -1e9

    if num_hard > 0:
        hard_score, _ = torch.topk(masked_score, k=num_hard, dim=1)
    else:
        hard_score = torch.empty(batch_size, 0, device=device)

    if num_rand > 0:
        rand_idx = torch.randint(0, num_entities, (batch_size, num_rand), device=device)
        mask = rand_idx == label.unsqueeze(1)
        while mask.any():
            rand_idx[mask] = torch.randint(0, num_entities, (mask.sum().item(),), device=device)
            mask = rand_idx == label.unsqueeze(1)
        rand_score = score.gather(1, rand_idx)
    else:
        rand_score = torch.empty(batch_size, 0, device=device)

    combined_score = torch.cat([pos_score.unsqueeze(1), hard_score, rand_score], dim=1)
    target = torch.zeros(batch_size, dtype=torch.long, device=device)
    return nn.CrossEntropyLoss(label_smoothing=smoothing)(combined_score, target)


def make_model_class(MyGO):
    class MyGORepro(MyGO):
        """MyGO của tác giả, chỉ thay bước rụng mô thức và BGF cho dễ bật tắt."""

        md_mode = "code"
        md_alpha = 0.3
        use_bgf = True

        def ModalityAwareFusion(self, rep_ent_str, rep_ent_vis, rep_ent_txt,
                                ent_vis_mask, ent_txt_mask, p_drop):
            if self.training and self.md_mode != "none":
                # Mã gốc: rand(1) < 0.3 cố định; bài báo: rand(1) < P_drop(epoch).
                p = self.md_alpha if self.md_mode == "code" else p_drop
                if torch.rand(1).item() < p:
                    if torch.rand(1).item() < 0.5:
                        rep_ent_txt = torch.zeros_like(rep_ent_txt)
                    else:
                        rep_ent_vis = torch.zeros_like(rep_ent_vis)
                    return rep_ent_str, rep_ent_vis, rep_ent_txt, None
            if self.use_bgf:
                vis_avg = rep_ent_vis.mean(1, keepdim=True)
                txt_avg = rep_ent_txt.mean(1, keepdim=True)
                vis_gate = self.vis2txt_gate(torch.cat([vis_avg, txt_avg], -1))
                txt_gate = self.txt2vis_gate(torch.cat([txt_avg, vis_avg], -1))
                rep_ent_vis = rep_ent_vis * vis_gate
                rep_ent_txt = rep_ent_txt * txt_gate
            # Mặt nạ trả về không được forward() dùng tới, nên trả None.
            return rep_ent_str, rep_ent_vis, rep_ent_txt, None

    return MyGORepro


@torch.no_grad()
def evaluate(model, KG, split, batch_size):
    """Xếp hạng có lọc cho cả hai chiều, giống utils.calculate_rank nhưng chạy theo lô."""
    model.eval()
    ent_embs, rel_embs = model()
    num_ent, num_rel = KG.num_ent, KG.num_rel
    triples = getattr(KG, split)
    ranks = []
    for head in (True, False):
        for i in range(0, len(triples), batch_size):
            chunk = triples[i:i + batch_size]
            h = torch.tensor([x[0] for x in chunk])
            r = torch.tensor([x[1] for x in chunk])
            t = torch.tensor([x[2] for x in chunk])
            mask_tok = torch.full_like(h, num_ent + num_rel)
            if head:
                query = torch.stack([mask_tok, r + num_ent, t + num_rel], dim=1)
                target = h
                keys = [(-1, b, c) for b, c in zip(r.tolist(), t.tolist())]
            else:
                query = torch.stack([h + num_rel, r + num_ent, mask_tok], dim=1)
                target = t
                keys = [(a, b, -1) for a, b in zip(h.tolist(), r.tolist())]
            score = model.score(ent_embs, rel_embs, query.cuda())
            target = target.cuda()
            rows = torch.arange(len(chunk), device=score.device)
            score_target = score[rows, target].clone()
            # Như mã gốc: mọi đáp án đúng đã biết (kể cả chính đáp án) bị hạ xuống dưới đáp án.
            fr, fc = [], []
            for j, key in enumerate(keys):
                cols = KG.filter_dict[key]
                fr.extend([j] * len(cols))
                fc.extend(cols)
            fr = torch.tensor(fr, device=score.device)
            fc = torch.tensor(fc, device=score.device)
            score[fr, fc] = score_target[fr] - 1
            st = score_target.unsqueeze(1)
            rank = (score > st).sum(1) + (score == st).sum(1) // 2 + 1
            ranks.append(rank.cpu())
    ranks = torch.cat(ranks).double()
    return {
        "MR": ranks.mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "Hits@10": (ranks <= 10).double().mean().item(),
        "Hits@3": (ranks <= 3).double().mean().item(),
        "Hits@1": (ranks <= 1).double().mean().item(),
    }


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, **payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    args.out = os.path.abspath(args.out)
    args.name = args.name or os.path.basename(args.out.rstrip("/"))
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "last.ckpt")

    # Mã tác giả đọc data/ và tokens/ theo đường dẫn tương đối.
    os.chdir(args.src)
    sys.path.insert(0, args.src)
    from dataset import VTKG
    from merge_tokens import get_entity_textual_tokens, get_entity_visual_tokens
    from model_mygo import MyGO
    from module import MultiGenerator

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    class _Log:
        def info(self, *a):
            pass

    KG = VTKG(args.data, _Log(), max_vis_len=args.max_img_num)
    loader = torch.utils.data.DataLoader(KG, batch_size=args.batch_size, shuffle=True)
    vis_idx, vis_mask = get_entity_visual_tokens(dataset=args.data, max_num=args.max_vis_token)
    txt_idx, txt_mask = get_entity_textual_tokens(dataset=args.data, max_num=args.max_txt_token)

    Model = make_model_class(MyGO)
    Model.md_mode = args.md
    Model.md_alpha = args.alpha
    Model.use_bgf = not args.no_bgf
    model = Model(
        num_ent=KG.num_ent, num_rel=KG.num_rel,
        ent_vis_mask=vis_mask, ent_txt_mask=txt_mask,
        dim_str=args.dim, num_head=args.num_head, dim_hid=args.hidden_dim,
        num_layer_enc_ent=args.num_layer_enc_ent, num_layer_enc_rel=args.num_layer_enc_rel,
        num_layer_dec=args.num_layer_dec, dropout=args.dropout,
        emb_dropout=args.emb_dropout, vis_dropout=args.vis_dropout, txt_dropout=args.txt_dropout,
        visual_token_index=vis_idx.cuda(), text_token_index=txt_idx.cuda(),
        score_function=args.score_function,
        adv_generator=MultiGenerator(noise_dim=64, structure_dim=args.dim, img_dim=args.dim),
    ).cuda()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, T_mult=2)

    start_epoch, wandb_id = 1, None
    best = {"by_valid": None, "by_test": None}
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        set_rng_state(ck["rng"])
        start_epoch, wandb_id, best = ck["epoch"] + 1, ck["wandb_id"], ck["best"]
        print(f"Chạy tiếp từ epoch {start_epoch}")

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.name,
                         config=vars(args), id=wandb_id, resume="allow", dir=args.out)
        wandb_id = run.id
        # Trục x là epoch: khi chạy tiếp, các epoch sau checkpoint được ghi lại mà không bị bỏ.
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")

    t0 = time.time()
    N = args.num_epoch
    for epoch in range(start_epoch, N + 1):
        model.train()
        t_ep = time.time()
        p_drop = min(args.alpha, epoch / N * args.alpha)
        hard_ratio = min(args.beta, epoch / N * args.beta)
        total_loss = 0.0
        for batch, label in loader:
            ent_embs, rel_embs = model(p_drop, gen=0)
            scores = model.score(ent_embs, rel_embs, batch.cuda())
            label = label.cuda()
            loss = loss_fn(scores, label)
            if args.mu != 0:
                loss = loss + args.mu * hybrid_negative_ce_loss(
                    scores, label, num_neg=args.num_neg, hard_ratio=hard_ratio,
                    smoothing=args.smoothing)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        log = {
            "epoch": epoch,
            "train/loss": total_loss,
            "train/p_drop": {"code": args.alpha, "paper": p_drop, "none": 0.0}[args.md],
            "train/hard_ratio": hard_ratio if args.mu != 0 else 0.0,
            "time/epoch_sec": time.time() - t_ep,
            "gpu/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        }

        out_of_time = args.max_hours > 0 and (time.time() - t0) / 3600 >= args.max_hours
        if epoch % args.valid_epoch == 0 or epoch == N:
            valid = evaluate(model, KG, "valid", args.eval_batch)
            test = evaluate(model, KG, "test", args.eval_batch)
            log.update({f"valid/{k}": v for k, v in valid.items()})
            log.update({f"test/{k}": v for k, v in test.items()})
            rec = {"epoch": epoch, "valid": valid, "test": test}
            if best["by_valid"] is None or valid["MRR"] > best["by_valid"]["valid"]["MRR"]:
                best["by_valid"] = rec
            if best["by_test"] is None or test["MRR"] > best["by_test"]["test"]["MRR"]:
                best["by_test"] = rec
            print(f"[{epoch}] loss {total_loss:.4f} | valid MRR {valid['MRR']:.4f} "
                  f"| test MRR {test['MRR']:.4f} | {log['time/epoch_sec']:.1f}s/epoch "
                  f"| GPU max {log['gpu/max_mem_gb']:.1f} GB", flush=True)
            if run is not None:
                run.summary["best_by_valid"] = best["by_valid"]
                run.summary["best_by_test"] = best["by_test"]

        if run is not None:
            run.log(log)

        if epoch % args.valid_epoch == 0 or epoch == N or out_of_time:
            save_checkpoint(ckpt_path, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), rng=rng_state(), epoch=epoch,
                            wandb_id=wandb_id, best=best)
        if out_of_time and epoch < N:
            print(f"Hết {args.max_hours} giờ, đã lưu checkpoint ở epoch {epoch}. Chạy lại lệnh để tiếp tục.")
            break

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
