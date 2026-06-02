"""predictor_e2e.py

端到端推理：整张公式图 -> token 序列 -> 拼接成 LaTeX/文本。

支持两种模型版本：
- V2: 原始模型 (model_e2e.py)
- V3: 优化版模型 (model_e2e_v3.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from inference.image_preprocessing import read_image_unicode
from train.dataset_e2e_v3 import load_vocab, _binarize_inv, _resize_keep_ratio


@dataclass
class PredictorE2EConfig:
    ckpt_path: str = "checkpoints_e2e/best.ckpt"
    vocab_path: str = "checkpoints_e2e/vocab.json"
    device: str = "cuda"
    img_height: int = 64
    max_width: int = 512
    min_short_fraction_width: int = 104
    max_len: int = 256
    beam_width: int = 5
    model_version: str = "auto"  # "v2", "v3", or "auto"


def _estimate_background_value(img: np.ndarray) -> int:
    border = np.concatenate([
        img[:, :1].reshape(-1),
        img[:, -1:].reshape(-1),
        img[:1, :].reshape(-1),
        img[-1:, :].reshape(-1),
    ])
    return 255 if float(border.mean()) > 127.0 else 0


def _ink_mask(img: np.ndarray) -> np.ndarray:
    background = _estimate_background_value(img)
    if background > 127:
        return img < 128
    return img > 128


def _has_fraction_bar(img: np.ndarray) -> bool:
    """Detect compact fraction layouts that need more horizontal context."""
    ink = _ink_mask(img)
    h, w = ink.shape[:2]
    ink_count = int(np.count_nonzero(ink))
    if ink_count < 24:
        return False

    min_side_ink = max(8, int(ink_count * 0.08))
    min_bar_width = max(24, int(w * 0.6))
    max_bar_height = max(5, int(h * 0.16))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    for label in range(1, num_labels):
        x, y, cw, ch, area = stats[label]
        if area < 8 or ch <= 0:
            continue
        above = int(np.count_nonzero(ink[:y, :]))
        below = int(np.count_nonzero(ink[y + ch:, :]))
        if (
            cw >= min_bar_width
            and ch <= max_bar_height
            and (cw / float(ch)) >= 5.0
            and h * 0.22 <= y <= h * 0.78
            and above >= min_side_ink
            and below >= min_side_ink
        ):
            return True

    row_counts = ink.sum(axis=1)
    high_rows = np.where(row_counts >= int(w * 0.58))[0]
    if high_rows.size == 0:
        return False

    runs: list[list[int]] = []
    current = [int(high_rows[0])]
    for row in high_rows[1:]:
        row = int(row)
        if row == current[-1] + 1:
            current.append(row)
        else:
            runs.append(current)
            current = [row]
    runs.append(current)

    for run in runs:
        y0, y1 = run[0], run[-1]
        if (y1 - y0 + 1) > max_bar_height:
            continue
        above = int(np.count_nonzero(ink[:y0, :]))
        below = int(np.count_nonzero(ink[y1 + 1:, :]))
        if h * 0.22 <= y0 <= h * 0.78 and above >= min_side_ink and below >= min_side_ink:
            return True

    return False


def _pad_short_fraction_if_needed(img: np.ndarray, min_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min_width <= 0 or w >= min_width:
        return img
    if not _has_fraction_bar(img):
        return img

    pad_total = min_width - w
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    background = _estimate_background_value(img)
    return cv2.copyMakeBorder(
        img,
        0,
        0,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=background,
    )


class E2EPredictor:
    def __init__(self, cfg: PredictorE2EConfig | None = None) -> None:
        self.cfg = cfg or PredictorE2EConfig()
        self.device = torch.device(self.cfg.device if (self.cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

        token2id, id2token = load_vocab(self.cfg.vocab_path)
        self.token2id = token2id
        self.id2token = id2token

        if r"\ln" not in token2id:
            print(
                "[warn] The loaded vocab is missing \\ln. "
                "Natural-log formulas will likely decode as <unk> until the model is retrained with an updated vocab."
            )

        self.pad_id = token2id["<pad>"]
        self.bos_id = token2id["<bos>"]
        self.eos_id = token2id["<eos>"]
        self.unk_id = token2id["<unk>"]

        ckpt_file = Path(self.cfg.ckpt_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"未找到模型：{ckpt_file.resolve()}。请先训练模型。"
            )

        ckpt = torch.load(str(ckpt_file), map_location=self.device, weights_only=False)
        model_cfg = ckpt.get("model_cfg")
        vocab_size = int(ckpt.get("vocab_size"))
        if model_cfg is None or vocab_size <= 0:
            raise RuntimeError("checkpoint 中缺少 model_cfg/vocab_size")

        # 自动检测模型版本
        model_version = self.cfg.model_version
        if model_version == "auto":
            # 根据model_cfg中的字段判断版本
            if "attention_dim" in model_cfg or model_cfg.get("decoder_layers", 2) > 2:
                model_version = "v3"
            else:
                model_version = "v2"
        
        self.model_version = model_version

        if model_version == "v3":
            from train.model_e2e_v3 import build_e2e_model_v3
            self.model = build_e2e_model_v3(vocab_size, model_cfg).to(self.device)
            print(f"使用优化版模型 V3")
        else:
            from train.model_e2e import build_e2e_model_v2
            self.model = build_e2e_model_v2(vocab_size, model_cfg).to(self.device)  # 这行缩进！
            print(f"使用原始模型 V2")
        
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        print(f"模型加载成功，设备: {self.device}")

    def _preprocess(self, img_path: str) -> torch.Tensor:
        gray = read_image_unicode(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"无法读取图片：{img_path}")

        x = _binarize_inv(gray)
        x = _resize_keep_ratio(x, self.cfg.img_height, self.cfg.max_width)
        x = _pad_short_fraction_if_needed(x, self.cfg.min_short_fraction_width)
        x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]
        return x_t.to(self.device)
    
    def _preprocess_numpy(self, img: np.ndarray) -> torch.Tensor:
        """从numpy数组预处理图像"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        x = _binarize_inv(gray)
        x = _resize_keep_ratio(x, self.cfg.img_height, self.cfg.max_width)
        x = _pad_short_fraction_if_needed(x, self.cfg.min_short_fraction_width)
        x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0) / 255.0
        return x_t.to(self.device)

    @torch.no_grad()
    def predict_tokens(self, img_path: str, use_beam_search: bool = True) -> List[str]:
        """预测图像中的公式token序列"""
        image = self._preprocess(img_path)
        
        if use_beam_search:
            preds = self.model.beam_search_decode(
                image, 
                bos_id=self.bos_id, 
                eos_id=self.eos_id, 
                max_len=self.cfg.max_len,
                beam_width=self.cfg.beam_width
            )
        else:
            preds = self.model.greedy_decode(
                image, 
                bos_id=self.bos_id, 
                eos_id=self.eos_id, 
                max_len=self.cfg.max_len
            )
        
        out_ids = preds[0].tolist()
        
        result = []
        for idx in out_ids:
            if idx == self.bos_id:
                continue
            if idx == self.eos_id:
                break
            if idx == self.pad_id:
                continue
            result.append(self.id2token.get(idx, "<unk>"))
        
        return result
    
    @torch.no_grad()
    def predict_tokens_from_numpy(self, img: np.ndarray, use_beam_search: bool = True) -> List[str]:
        """从numpy数组预测公式token序列"""
        image = self._preprocess_numpy(img)
        
        if use_beam_search:
            preds = self.model.beam_search_decode(
                image, 
                bos_id=self.bos_id, 
                eos_id=self.eos_id, 
                max_len=self.cfg.max_len,
                beam_width=self.cfg.beam_width
            )
        else:
            preds = self.model.greedy_decode(
                image, 
                bos_id=self.bos_id, 
                eos_id=self.eos_id, 
                max_len=self.cfg.max_len
            )
        
        out_ids = preds[0].tolist()
        
        result = []
        for idx in out_ids:
            if idx == self.bos_id:
                continue
            if idx == self.eos_id:
                break
            if idx == self.pad_id:
                continue
            result.append(self.id2token.get(idx, "<unk>"))
        
        return result

    def predict_text(self, img_path: str, use_beam_search: bool = True) -> str:
        """预测并返回公式文本"""
        toks = self.predict_tokens(img_path, use_beam_search=use_beam_search)
        return " ".join(toks)
    
    def predict_text_from_numpy(self, img: np.ndarray, use_beam_search: bool = True) -> str:
        """从numpy数组预测并返回公式文本"""
        toks = self.predict_tokens_from_numpy(img, use_beam_search=use_beam_search)
        return " ".join(toks)


def create_predictor(device: str = "cuda", use_v3: bool = False) -> E2EPredictor:
    """创建预测器
    
    Args:
        device: 设备 ("cuda" 或 "cpu")
        use_v3: 是否优先使用V3模型
    """
    # 优先检查V3模型
    if use_v3:
        ckpt_v3 = Path("checkpoints_e2e_v3/best.ckpt")
        vocab_v3 = Path("checkpoints_e2e_v3/vocab.json")
        if ckpt_v3.exists() and vocab_v3.exists():
            cfg = PredictorE2EConfig(
                ckpt_path=str(ckpt_v3),
                vocab_path=str(vocab_v3),
                device=device,
                model_version="v3",
            )
            return E2EPredictor(cfg)
    
    # 检查V2模型
    ckpt = Path("checkpoints_e2e/best.ckpt")
    vocab = Path("checkpoints_e2e/vocab.json")
    
    if not ckpt.exists() or not vocab.exists():
        # 再次检查V3
        ckpt_v3 = Path("checkpoints_e2e_v3/best.ckpt")
        vocab_v3 = Path("checkpoints_e2e_v3/vocab.json")
        if ckpt_v3.exists() and vocab_v3.exists():
            cfg = PredictorE2EConfig(
                ckpt_path=str(ckpt_v3),
                vocab_path=str(vocab_v3),
                device=device,
                model_version="v3",
            )
            return E2EPredictor(cfg)
        
        raise FileNotFoundError(
            "未找到训练好的模型，请先运行训练：\n"
            "python -m train.train_e2e_v3 --data_root \"D:/CROHME/CROHME\" --epochs 200 --batch_size 16 --device cuda --amp"
        )
    
    cfg = PredictorE2EConfig(
        ckpt_path=str(ckpt),
        vocab_path=str(vocab),
        device=device,
        model_version="auto",
    )
    
    return E2EPredictor(cfg)


def create_predictor_v3(device: str = "cuda") -> E2EPredictor:
    """创建V3版本预测器"""
    ckpt = Path("checkpoints_e2e_v3/best.ckpt")
    vocab = Path("checkpoints_e2e_v3/vocab.json")
    
    if not ckpt.exists() or not vocab.exists():
        raise FileNotFoundError(
            "未找到V3模型，请先运行训练：\n"
            "python -m train.train_e2e_v3 --data_root \"D:/CROHME/CROHME\" --epochs 200 --batch_size 16 --device cuda --amp"
        )
    
    cfg = PredictorE2EConfig(
        ckpt_path=str(ckpt),
        vocab_path=str(vocab),
        device=device,
        model_version="v3",
    )
    
    return E2EPredictor(cfg)
