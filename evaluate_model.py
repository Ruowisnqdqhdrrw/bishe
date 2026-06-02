"""evaluate_model.py

与 train_e2e_v3.py 完全一致的评估脚本

核心原则：
  - decode 调用与训练 sequence_exact_match 完全相同
  - vocab 必须来自 checkpoint 同目录，且 size 必须匹配
  - 数据集配置从 checkpoint["cfg"] 恢复，保证 split 一致
  - 模型构建参数从 checkpoint["model_cfg"] 恢复

用法：
  # 完整评估
  python evaluate_model.py --checkpoint checkpoints_e2e_v3/best.ckpt

  # Debug 模式（只跑前 100 个样本）
  python evaluate_model.py --checkpoint checkpoints_e2e_v3/best.ckpt --debug 100

  # Greedy decode
  python evaluate_model.py --checkpoint checkpoints_e2e_v3/best.ckpt --no_beam_search

  # 覆盖数据路径（checkpoint 中的路径不可用时）
  python evaluate_model.py --checkpoint checkpoints_e2e_v3/best.ckpt --data_root D:/CROHME/CROHME
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.dataset_e2e_v3 import (
    E2EDatasetConfigV3,
    E2EFormulaDatasetV3,
    build_e2e_pairs,
    collate_e2e_v3,
    load_vocab,
    split_train_val_pairs,
)
from train.model_e2e_v3 import build_e2e_model_v3
from train.utils import set_random_seed


# ------------------------------------------------------------------ #
#  1. Checkpoint 加载 + 完整性校验
# ------------------------------------------------------------------ #

def load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    required = ["model", "vocab_size", "pad_id", "model_cfg", "cfg"]
    missing = [f for f in required if f not in ckpt]
    if missing:
        raise ValueError(f"Checkpoint missing fields: {missing}")

    print(f"\n{'='*60}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"  epoch          : {ckpt.get('epoch', 'N/A')}")
    print(f"  vocab_size     : {ckpt['vocab_size']}")
    print(f"  pad_id         : {ckpt['pad_id']}")
    print(f"  best_val_seq   : {ckpt.get('best_val_seq_acc', 0):.4f}")
    print(f"  best_val_tok   : {ckpt.get('best_val_tok_acc', 0):.4f}")
    print(f"  model_cfg      : {ckpt['model_cfg']}")
    print(f"{'='*60}")

    return ckpt


# ------------------------------------------------------------------ #
#  2. Vocab 加载 + 一致性校验
# ------------------------------------------------------------------ #

def load_vocab_strict(ckpt_path: Path, ckpt: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """只允许从 checkpoint 同目录加载 vocab，并严格校验 size。"""
    vocab_path = ckpt_path.parent / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"vocab.json not found at {vocab_path}\n"
            f"Vocab MUST be in the same directory as the checkpoint."
        )

    token2id, id2token = load_vocab(str(vocab_path))

    # ---- 严格校验 ----
    assert len(token2id) == ckpt["vocab_size"], (
        f"Vocab size mismatch! "
        f"checkpoint expects {ckpt['vocab_size']}, loaded {len(token2id)}"
    )
    assert token2id.get("<pad>") == ckpt["pad_id"], (
        f"pad_id mismatch! "
        f"checkpoint={ckpt['pad_id']}, vocab={token2id.get('<pad>')}"
    )

    print(f"\nVocab      : {vocab_path}")
    print(f"  size           : {len(token2id)}")
    print(f"  pad={token2id['<pad>']}  bos={token2id['<bos>']}  eos={token2id['<eos>']}")
    print(f"  consistency    : OK")

    return token2id, id2token


# ------------------------------------------------------------------ #
#  3. 数据集构建（完全复用 checkpoint 中的 cfg）
# ------------------------------------------------------------------ #

def build_val_loader(
    ckpt: dict,
    token2id: Dict[str, int],
    batch_size: int = 16,
    data_root_override: Optional[str] = None,
) -> Tuple[DataLoader, list]:
    """用 checkpoint 保存的 cfg 构建验证集，保证 split 与训练一致。"""

    cfg_dict = dict(ckpt["cfg"])  # shallow copy

    # asdict 会把 tuple 变成 list，这里转回 tuple
    if "splits" in cfg_dict and isinstance(cfg_dict["splits"], list):
        cfg_dict["splits"] = tuple(cfg_dict["splits"])

    # 允许覆盖 data_root（换机器时路径可能不同）
    if data_root_override:
        cfg_dict["data_root"] = data_root_override

    cfg = E2EDatasetConfigV3(**cfg_dict)

    # 与训练脚本完全相同的流程
    set_random_seed(cfg.seed)
    pairs = build_e2e_pairs(cfg)
    if len(pairs) == 0:
        raise ValueError(f"No data found under {cfg.data_root}, splits={cfg.splits}")

    train_pairs, val_pairs = split_train_val_pairs(pairs, cfg.val_ratio, cfg.seed)

    print(f"\nDataset")
    print(f"  data_root      : {cfg.data_root}")
    print(f"  splits         : {cfg.splits}")
    print(f"  val_ratio      : {cfg.val_ratio}")
    print(f"  seed           : {cfg.seed}")
    print(f"  augment_level  : {cfg.augment_level}")
    print(f"  total pairs    : {len(pairs)}")
    print(f"  train          : {len(train_pairs)}")
    print(f"  val            : {len(val_pairs)}")

    val_ds = E2EFormulaDatasetV3(val_pairs, token2id, cfg, augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_e2e_v3,
    )

    return val_loader, val_pairs


# ------------------------------------------------------------------ #
#  4. 模型构建（严格对齐 checkpoint）
# ------------------------------------------------------------------ #

def build_model(ckpt: dict, vocab_size: int, device: torch.device):
    model_cfg = ckpt["model_cfg"]

    # 严格校验
    assert ckpt["vocab_size"] == vocab_size, (
        f"vocab_size mismatch: ckpt={ckpt['vocab_size']} vs loaded={vocab_size}"
    )

    model = build_e2e_model_v3(vocab_size, model_cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel")
    print(f"  params         : {n_params / 1e6:.2f}M")
    print(f"  vocab_size     : {vocab_size}")
    print(f"  model_cfg      : {model_cfg}")

    return model


# ------------------------------------------------------------------ #
#  5. 评估（decode 与 train_e2e_v3.sequence_exact_match 完全一致）
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    use_beam_search: bool = True,
    beam_width: int = 10,
    limit_batches: Optional[int] = None,
    id2token: Optional[Dict[int, str]] = None,
    show_examples: int = 10,
) -> Tuple[float, float]:
    """
    与 train_e2e_v3.py 中 sequence_exact_match 完全一致的评估逻辑。

    关键：beam_search_decode 只传 (images, bos_id, eos_id, max_len, beam_width)，
    不额外传 length_penalty / coverage_penalty / repetition_penalty，
    让模型使用自身默认值——与训练验证时完全相同。
    """
    model.eval()
    n_samples = 0
    n_correct_seq = 0
    n_correct_tok = 0
    n_total_tok = 0
    examples_shown = 0

    for bi, (images, _widths, targets) in enumerate(tqdm(loader, desc="Eval")):
        if limit_batches is not None and bi >= limit_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ---------- decode：与训练完全一致 ----------
        if use_beam_search:
            preds = model.beam_search_decode(
                images,
                bos_id=bos_id,
                eos_id=eos_id,
                max_len=max_len,
                beam_width=beam_width,
            )
        else:
            preds = model.greedy_decode(
                images,
                bos_id=bos_id,
                eos_id=eos_id,
                max_len=max_len,
            )

        # ---------- strip_to_eos：与训练完全一致 ----------
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

        # ---------- 对齐长度：与训练完全一致 ----------
        t_pred = preds2.size(1)
        t_tgt = targets2.size(1)
        if t_pred > t_tgt:
            pad = torch.full((targets2.size(0), t_pred - t_tgt), pad_id,
                             dtype=targets2.dtype, device=device)
            targets2 = torch.cat([targets2, pad], dim=1)
        elif t_tgt > t_pred:
            pad = torch.full((preds2.size(0), t_tgt - t_pred), pad_id,
                             dtype=preds2.dtype, device=device)
            preds2 = torch.cat([preds2, pad], dim=1)

        # ---------- 统计：与训练完全一致 ----------
        eq = (preds2 == targets2)
        is_ok = eq.all(dim=1)
        n_correct_seq += int(is_ok.sum().item())
        n_samples += int(is_ok.numel())

        mask = (targets2 != pad_id) & (targets2 != bos_id)
        n_correct_tok += int(((preds2 == targets2) & mask).sum().item())
        n_total_tok += int(mask.sum().item())

        # ---------- debug 输出前 N 条 ----------
        if show_examples > 0 and examples_shown < show_examples and id2token is not None:
            for i in range(min(images.size(0), show_examples - examples_shown)):
                gt_ids = targets2[i].cpu().tolist()
                pr_ids = preds2[i].cpu().tolist()

                gt_tok = [id2token.get(t, f"<{t}>") for t in gt_ids
                          if t != pad_id and t != bos_id]
                pr_tok = [id2token.get(t, f"<{t}>") for t in pr_ids
                          if t != pad_id and t != bos_id]

                tag = "OK" if is_ok[i].item() else "FAIL"
                print(f"\n  [{tag}] sample {examples_shown + 1}")
                print(f"    GT  : {' '.join(gt_tok)}")
                print(f"    PRED: {' '.join(pr_tok)}")
                examples_shown += 1
                if examples_shown >= show_examples:
                    break

    seq_acc = n_correct_seq / max(1, n_samples)
    tok_acc = n_correct_tok / max(1, n_total_tok)
    return seq_acc, tok_acc


# ------------------------------------------------------------------ #
#  main
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="评估脚本（与 train_e2e_v3 完全一致）")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_beam_search", action="store_true",
                    help="使用 greedy decode")
    ap.add_argument("--beam_width", type=int, default=10,
                    help="beam search 宽度（训练默认 10）")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--debug", type=int, default=None,
                    help="只评估前 N 个 batch（快速 sanity check）")
    ap.add_argument("--show_examples", type=int, default=10,
                    help="打印前 N 条 GT vs PRED")
    ap.add_argument("--data_root", type=str, default=None,
                    help="覆盖 checkpoint 中的 data_root 路径")
    args = ap.parse_args()

    # ---- device ----
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ---- 1. checkpoint ----
    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, device)

    # ---- 2. vocab（严格校验） ----
    token2id, id2token = load_vocab_strict(ckpt_path, ckpt)
    vocab_size = len(token2id)
    pad_id = token2id["<pad>"]
    bos_id = token2id["<bos>"]
    eos_id = token2id["<eos>"]

    # ---- 3. dataset（从 checkpoint cfg 恢复） ----
    val_loader, val_pairs = build_val_loader(
        ckpt, token2id,
        batch_size=args.batch_size,
        data_root_override=args.data_root,
    )

    # ---- 4. model（严格校验） ----
    model = build_model(ckpt, vocab_size, device)

    # ---- 5. 汇总信息 ----
    max_len = ckpt["cfg"]["max_seq_len"]
    use_beam = not args.no_beam_search

    print(f"\n{'='*60}")
    print(f"Evaluation")
    print(f"  checkpoint vocab_size : {ckpt['vocab_size']}")
    print(f"  loaded vocab_size     : {vocab_size}")
    print(f"  val samples           : {len(val_pairs)}")
    print(f"  decode                : {'beam (w=' + str(args.beam_width) + ')' if use_beam else 'greedy'}")
    print(f"  max_len               : {max_len}")
    if args.debug:
        print(f"  debug batches         : {args.debug}")
    print(f"{'='*60}\n")

    # ---- 6. evaluate ----
    seq_acc, tok_acc = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_len=max_len,
        use_beam_search=use_beam,
        beam_width=args.beam_width,
        limit_batches=args.debug,
        id2token=id2token,
        show_examples=args.show_examples,
    )

    # ---- 7. 结果 ----
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Sequence Accuracy : {seq_acc * 100:.2f}%")
    print(f"  Token Accuracy    : {tok_acc * 100:.2f}%")
    print(f"{'='*60}")

    # ---- 8. 与训练对比 ----
    train_best = ckpt.get("best_val_seq_acc", 0)
    if train_best > 0:
        diff = seq_acc - train_best
        print(f"\n  Training best seq acc : {train_best * 100:.2f}%")
        print(f"  This eval seq acc     : {seq_acc * 100:.2f}%")
        print(f"  Difference            : {diff * 100:+.2f}%")

        if abs(diff) > 0.05:
            print(f"\n  [WARNING] Gap > 5%. Possible causes:")
            print(f"    - vocab.json 与训练时不同")
            print(f"    - data_root 路径下数据文件变化")
            print(f"    - splits / val_ratio / seed 不一致")
            print(f"    - beam_width 不同（训练默认 10）")
        else:
            print(f"\n  [OK] 与训练结果一致")

    print()


if __name__ == "__main__":
    main()
