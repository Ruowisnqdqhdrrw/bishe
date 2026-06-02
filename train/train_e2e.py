"""train_e2e.py

端到端公式识别训练脚本 V3：
1. 使用增强版模型（残差CNN + SE注意力 + DropPath + 覆盖注意力 + LSTM）
2. 强数据增强
3. Cosine退火学习率、早停、梯度累积
4. 覆盖损失防止重复关注
5. Focal Loss处理类别不平衡

目标：单字符准确率>=96%，公式准确率>=92%

用法：
python -m train.train_e2e --data_root "D:/CROHME/CROHME" --epochs 200 --batch_size 20 --device cuda --amp

推荐参数（RTX 5060）：
python -m train.train_e2e --data_root "D:/CROHME/CROHME" --epochs 200 --batch_size 20 --lr 8e-4 --device cuda --amp --grad_accum 2 --patience 30 --num_workers 0 --dropout 0.5 --drop_path 0.15
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.dataset_e2e import (
    E2EDatasetConfigV2,
    E2EFormulaDatasetV2,
    build_e2e_pairs,
    build_vocab_from_captions,
    collate_e2e_v2,
    load_vocab,
    save_vocab,
    split_train_val_pairs,
)
from train.model_e2e import build_e2e_model_v2
from train.utils import set_random_seed


def save_checkpoint(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    return torch.load(str(path), map_location=device, weights_only=False)


def ensure_vocab_covers_pairs(
    *,
    pairs,
    token2id: Dict[str, int],
    vocab_path: Path,
    resume_requested: bool,
    resume_ckpt_path: Path,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    current_token2id, _ = build_vocab_from_captions(pairs, min_freq=1)
    missing_tokens = [tok for tok in current_token2id.keys() if tok not in token2id]
    if not missing_tokens:
        return token2id, {i: t for t, i in token2id.items()}

    preview = ", ".join(missing_tokens[:12])
    if len(missing_tokens) > 12:
        preview += ", ..."

    if resume_requested and resume_ckpt_path.exists():
        raise SystemExit(
            "Existing vocab is missing caption tokens and cannot be expanded while resuming. "
            f"Missing tokens: {preview}. Start a fresh run without --resume."
        )

    updated = dict(token2id)
    for tok in missing_tokens:
        updated[tok] = len(updated)
    save_vocab(str(vocab_path), updated)
    print(f"[vocab] Added {len(missing_tokens)} new token(s) to {vocab_path.name}: {preview}")
    return updated, {i: t for t, i in updated.items()}


class FocalLoss(nn.Module):
    """Focal Loss：处理类别不平衡问题"""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int = 0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        mask = targets != self.ignore_index
        return focal_loss[mask].mean() if mask.sum() > 0 else focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing交叉熵损失"""
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        mask = targets != self.ignore_index
        loss = loss * mask.float()
        
        return loss.sum() / mask.sum().clamp(min=1)


def seq_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float = 0.1,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """序列交叉熵损失"""
    logits = logits[:, 1:, :].contiguous()
    targets = targets[:, 1:].contiguous()
    
    if use_focal:
        loss_fn = FocalLoss(gamma=focal_gamma, ignore_index=pad_id)
        return loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
    elif label_smoothing > 0:
        loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing, ignore_index=pad_id)
        return loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        return loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    """Token级准确率"""
    preds = logits.argmax(dim=-1)
    preds = preds[:, 1:]
    t = targets[:, 1:]
    mask = t != pad_id
    if mask.sum().item() == 0:
        return 0.0
    correct = ((preds == t) & mask).sum().item()
    total = mask.sum().item()
    return correct / total


@torch.no_grad()
def sequence_exact_match(
    model,
    loader,
    device,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    use_beam_search: bool = False,
    beam_width: int = 5,
    limit_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """计算序列完全匹配率和token准确率（使用greedy或beam search）"""
    model.eval()
    n_samples = 0
    n_correct_seq = 0
    n_correct_tok = 0
    n_total_tok = 0
    
    for bi, (images, _widths, targets) in enumerate(tqdm(loader, desc="Eval", leave=False)):
        if limit_batches is not None and bi >= limit_batches:
            break
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if use_beam_search:
            preds = model.beam_search_decode(images, bos_id=bos_id, eos_id=eos_id, 
                                             max_len=max_len, beam_width=beam_width)
        else:
            preds = model.greedy_decode(images, bos_id=bos_id, eos_id=eos_id, max_len=max_len)
        
        # 对齐长度
        def strip_to_eos(x: torch.Tensor) -> torch.Tensor:
            out = x.clone()
            for i in range(out.size(0)):
                row = out[i]
                eos_pos = (row == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    p = int(eos_pos[0].item())
                    if p + 1 < row.numel():
                        row[p + 1:] = pad_id
                out[i] = row
            return out
        
        preds2 = strip_to_eos(preds)
        targets2 = strip_to_eos(targets)
        
        # 对齐长度
        t_pred = preds2.size(1)
        t_tgt = targets2.size(1)
        if t_pred > t_tgt:
            pad = torch.full((targets2.size(0), t_pred - t_tgt), pad_id, dtype=targets2.dtype, device=device)
            targets2 = torch.cat([targets2, pad], dim=1)
        elif t_tgt > t_pred:
            pad = torch.full((preds2.size(0), t_tgt - t_pred), pad_id, dtype=preds2.dtype, device=device)
            preds2 = torch.cat([preds2, pad], dim=1)
        
        # 序列完全匹配
        eq = (preds2 == targets2)
        is_ok = eq.all(dim=1)
        n_correct_seq += int(is_ok.sum().item())
        n_samples += int(is_ok.numel())
        
        # Token准确率（排除padding和bos）
        mask = (targets2 != pad_id) & (targets2 != bos_id)
        n_correct_tok += int(((preds2 == targets2) & mask).sum().item())
        n_total_tok += int(mask.sum().item())
    
    seq_acc = n_correct_seq / max(1, n_samples)
    tok_acc = n_correct_tok / max(1, n_total_tok)
    return seq_acc, tok_acc


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    pad_id,
    tf_ratio: float,
    grad_clip: float,
    label_smoothing: float,
    use_amp: bool,
    scaler,
    grad_accum: int = 1,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for i, (images, _widths, tokens) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                result = model(images, tokens, teacher_forcing_ratio=tf_ratio)
                # 支持返回覆盖损失
                if isinstance(result, tuple):
                    logits, coverage_loss = result
                    ce_loss = seq_ce_loss(logits, tokens, pad_id, label_smoothing, use_focal, focal_gamma)
                    loss = ce_loss + coverage_loss
                else:
                    logits = result
                    loss = seq_ce_loss(logits, tokens, pad_id, label_smoothing, use_focal, focal_gamma)
            loss = loss / grad_accum
            scaler.scale(loss).backward()
        else:
            result = model(images, tokens, teacher_forcing_ratio=tf_ratio)
            if isinstance(result, tuple):
                logits, coverage_loss = result
                ce_loss = seq_ce_loss(logits, tokens, pad_id, label_smoothing, use_focal, focal_gamma)
                loss = ce_loss + coverage_loss
            else:
                logits = result
                loss = seq_ce_loss(logits, tokens, pad_id, label_smoothing, use_focal, focal_gamma)
            loss = loss / grad_accum
            loss.backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            if use_amp and device.type == "cuda":
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        acc = token_accuracy(logits.detach(), tokens, pad_id)
        total_acc += acc
        n += 1
        
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}", "acc": f"{acc:.4f}"})

    return total_loss / max(1, n), total_acc / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, pad_id, label_smoothing: float) -> Tuple[float, float]:
    """验证集评估（使用teacher forcing=0）"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for images, _widths, tokens in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            result = model(images, tokens, teacher_forcing_ratio=0.0)
            if isinstance(result, tuple):
                logits, _ = result
            else:
                logits = result
            loss = seq_ce_loss(logits, tokens, pad_id, label_smoothing=0.0)

        total_loss += loss.item()
        total_acc += token_accuracy(logits, tokens, pad_id)
        n += 1

    return total_loss / max(1, n), total_acc / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="D:/CROHME/CROHME")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    # 图像
    ap.add_argument("--img_height", type=int, default=64)
    ap.add_argument("--max_width", type=int, default=512)
    ap.add_argument("--max_seq_len", type=int, default=256)

    # 模型架构
    ap.add_argument("--encoder_dim", type=int, default=512)
    ap.add_argument("--decoder_dim", type=int, default=512)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--encoder_dropout", type=float, default=0.25)
    ap.add_argument("--drop_path", type=float, default=0.15, help="DropPath正则化率")
    ap.add_argument("--coverage_weight", type=float, default=0.1, help="覆盖损失权重")

    # 训练策略
    ap.add_argument("--teacher_forcing", type=float, default=0.7)
    ap.add_argument("--use_focal", action="store_true", help="使用Focal Loss")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma参数")
    ap.add_argument("--tf_decay", type=float, default=0.99, help="每epoch teacher forcing衰减率")
    ap.add_argument("--tf_min", type=float, default=0.4, help="teacher forcing最小值")
    ap.add_argument("--grad_clip", type=float, default=2.0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=2)
    
    # 学习率调度
    ap.add_argument("--scheduler", choices=["cosine", "onecycle", "none"], default="cosine")
    ap.add_argument("--warmup_epochs", type=int, default=5)
    
    # 早停
    ap.add_argument("--patience", type=int, default=25, help="早停耐心值")
    
    # 数据增强
    ap.add_argument("--augment_level", choices=["none", "light", "strong"], default="strong")
    
    # 其他
    ap.add_argument("--amp", action="store_true", help="使用混合精度训练")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval_beam", action="store_true", help="验证时使用beam search")
    ap.add_argument("--beam_width", type=int, default=5)

    args = ap.parse_args()

    set_random_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")

    cfg = E2EDatasetConfigV2(
        data_root=args.data_root,
        img_height=args.img_height,
        max_width=args.max_width,
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
        augment_level=args.augment_level,
    )

    # 1) 构建数据对
    pairs = build_e2e_pairs(cfg)
    if len(pairs) == 0:
        raise SystemExit("未读取到任何样本对，请检查数据路径")

    train_pairs, val_pairs = split_train_val_pairs(pairs, cfg.val_ratio, cfg.seed)
    print(f"读取公式样本：{len(pairs)}；训练：{len(train_pairs)}；验证：{len(val_pairs)}")

    # 2) 词汇表
    out_dir = Path("checkpoints_e2e")
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"
    last_ckpt = out_dir / "last.ckpt"

    if vocab_path.exists():
        token2id, id2token = load_vocab(str(vocab_path))
    else:
        token2id, id2token = build_vocab_from_captions(pairs, min_freq=1)
        save_vocab(str(vocab_path), token2id)

    token2id, id2token = ensure_vocab_covers_pairs(
        pairs=pairs,
        token2id=token2id,
        vocab_path=vocab_path,
        resume_requested=args.resume,
        resume_ckpt_path=last_ckpt,
    )

    vocab_size = len(token2id)
    pad_id = token2id["<pad>"]
    bos_id = token2id["<bos>"]
    eos_id = token2id["<eos>"]
    print(f"vocab_size={vocab_size} (pad_id={pad_id}, bos_id={bos_id}, eos_id={eos_id})")

    # 3) 数据集
    train_ds = E2EFormulaDatasetV2(train_pairs, token2id, cfg, augment=True)
    val_ds = E2EFormulaDatasetV2(val_pairs, token2id, cfg, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_e2e_v2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_e2e_v2,
    )

    # 4) 模型
    model_cfg = {
        "encoder_dim": args.encoder_dim,
        "decoder_dim": args.decoder_dim,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "encoder_dropout": args.encoder_dropout,
        "drop_path": args.drop_path,
        "coverage_weight": args.coverage_weight,
    }
    model = build_e2e_model_v2(vocab_size, model_cfg).to(device)
    
    # 打印模型参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    # 5) 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # 6) 学习率调度器
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    
    if args.scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(args.min_lr / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    else:
        scheduler = None

    # 7) 混合精度
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')

    # 8) 检查点
    best_ckpt = out_dir / "best.ckpt"
    start_epoch = 1
    best_val_seq_acc = 0.0
    best_val_tok_acc = 0.0
    tf_ratio = args.teacher_forcing

    if args.resume and last_ckpt.exists():
        ckpt = load_checkpoint(last_ckpt, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_seq_acc = float(ckpt.get("best_val_seq_acc", 0.0))
        best_val_tok_acc = float(ckpt.get("best_val_tok_acc", 0.0))
        tf_ratio = float(ckpt.get("tf_ratio", args.teacher_forcing))
        print(f"已恢复：epoch={start_epoch} best_val_seq_acc={best_val_seq_acc:.4f} best_val_tok_acc={best_val_tok_acc:.4f}")

    # 9) 早停
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode="max")

    # 10) 日志
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "train_e2e_log.csv"

    write_header = not csv_path.exists() or (not args.resume)
    
    import math
    
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch", "train_loss", "train_tok_acc", 
                "val_loss", "val_tok_acc", "val_seq_acc_greedy", "val_tok_acc_greedy",
                "lr", "tf_ratio", "time"
            ])

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()

            # 训练
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                pad_id=pad_id,
                tf_ratio=tf_ratio,
                grad_clip=args.grad_clip,
                label_smoothing=args.label_smoothing,
                use_amp=args.amp and device.type == 'cuda',
                scaler=scaler,
                grad_accum=args.grad_accum,
                use_focal=args.use_focal,
                focal_gamma=args.focal_gamma,
            )
            
            # 验证（teacher forcing=0）
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                pad_id=pad_id,
                label_smoothing=0.0,
            )

            # Greedy解码评估
            val_seq_acc, val_tok_acc_greedy = sequence_exact_match(
                model=model,
                loader=val_loader,
                device=device,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                max_len=args.max_seq_len,
                use_beam_search=args.eval_beam,
                beam_width=args.beam_width,
                limit_batches=None,  # 评估全部
            )

            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            t1 = time.time()

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} tok_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} tok_acc={val_acc:.4f} | "
                f"greedy: seq_acc={val_seq_acc:.4f} tok_acc={val_tok_acc_greedy:.4f} | "
                f"lr={lr:.2e} tf={tf_ratio:.2f} | time={t1-t0:.1f}s"
            )

            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", 
                f"{val_seq_acc:.6f}", f"{val_tok_acc_greedy:.6f}",
                f"{lr:.8f}", f"{tf_ratio:.4f}", f"{t1-t0:.1f}"
            ])
            f.flush()

            # 保存检查点
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "best_val_seq_acc": best_val_seq_acc,
                "best_val_tok_acc": best_val_tok_acc,
                "tf_ratio": tf_ratio,
                "cfg": asdict(cfg),
                "model_cfg": model_cfg,
                "vocab_size": vocab_size,
                "pad_id": pad_id,
            }
            save_checkpoint(last_ckpt, state)

            # 保存最佳模型（基于序列准确率）
            if val_seq_acc > best_val_seq_acc:
                best_val_seq_acc = val_seq_acc
                best_val_tok_acc = val_tok_acc_greedy
                state["best_val_seq_acc"] = best_val_seq_acc
                state["best_val_tok_acc"] = best_val_tok_acc
                save_checkpoint(best_ckpt, state)
                print(f"  ★ 新最佳模型！seq_acc={best_val_seq_acc:.4f} tok_acc={best_val_tok_acc:.4f}")

            # Teacher forcing衰减
            tf_ratio = max(args.tf_min, tf_ratio * args.tf_decay)

            # 早停检查
            if early_stopping(val_seq_acc):
                print(f"早停触发！在epoch {epoch}，最佳seq_acc={best_val_seq_acc:.4f}")
                break

    print(f"\n训练完成！")
    print(f"最佳验证序列准确率: {best_val_seq_acc:.4f}")
    print(f"最佳验证Token准确率: {best_val_tok_acc:.4f}")
    print(f"best: {best_ckpt.resolve()}")
    print(f"last: {last_ckpt.resolve()}")


if __name__ == "__main__":
    main()
