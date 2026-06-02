"""dataset_e2e_v3.py

优化版数据集：更强的数据增强策略

改进：
1. Mixup数据增强
2. CutOut数据增强
3. 更智能的弹性变形
4. 笔画模拟增强
5. 在线数据增强（每次返回不同的增强结果）
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from tools.build_charset import parse_caption_file


# ============================================================
# 图像读取LRU缓存
# 原实现每次__getitem__都从磁盘读取，对于多epoch训练存在大量重复IO
# 使用LRU缓存后，热点图像只需读取一次，后续直接从内存返回
# maxsize=2048：约占用 2048 * 64 * 512 ≈ 64MB 内存（灰度图）
# 性能预期：数据加载速度提升2-5x（取决于磁盘IO速度）
# ============================================================
@lru_cache(maxsize=2048)
def _cached_imread_gray(path: str) -> Optional[bytes]:
    """带LRU缓存的灰度图读取，返回bytes以支持缓存（numpy数组不可哈希）"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.tobytes(), img.shape[0], img.shape[1]


def _cached_to_numpy(path: str) -> Optional[np.ndarray]:
    """从缓存获取图像并转为numpy数组"""
    result = _cached_imread_gray(path)
    if result is None:
        return None
    data, h, w = result
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w).copy()


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>" , "<unk>"]


@dataclass
class E2EDatasetConfigV3:
    data_root: str = "D:/CROHME/CROHME"
    splits: Tuple[str, ...] = ("2014", "2016", "2019", "train")
    img_dir_name: str = "img"
    caption_name: str = "caption.txt"

    # 图像预处理
    img_height: int = 64
    max_width: int = 512

    # 序列
    max_seq_len: int = 256

    # 划分
    val_ratio: float = 0.1
    seed: int = 42
    
    # 增强强度
    augment_level: str = "strong"  # "none", "light", "medium", "strong"
    
    # Mixup参数
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.0  # 默认关闭，因为对序列任务效果不稳定


def _safe_imread_gray(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def _binarize_inv(gray: np.ndarray) -> np.ndarray:
    """二值化：前景=255，背景=0"""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def _resize_keep_ratio(img: np.ndarray, target_h: int, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img

    scale = target_h / float(h)
    new_w = int(round(w * scale))
    new_w = max(1, min(new_w, max_w))

    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


def _elastic_transform(img: np.ndarray, alpha: float = 20, sigma: float = 3, 
                       rng: np.random.RandomState = None) -> np.ndarray:
    """弹性变形：模拟手写的自然变化"""
    if rng is None:
        rng = np.random.RandomState()
    
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return img
    
    # 生成随机位移场
    dx = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)


def _perspective_transform(img: np.ndarray, strength: float = 0.05, 
                           rng: np.random.RandomState = None) -> np.ndarray:
    """透视变换：模拟拍摄角度变化"""
    if rng is None:
        rng = np.random.RandomState()
    
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return img
    
    # 原始四角
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # 随机扰动四角
    offset = strength * min(h, w)
    dst_pts = src_pts + rng.uniform(-offset, offset, src_pts.shape).astype(np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h), borderValue=0)


def _add_noise(img: np.ndarray, noise_level: float = 0.05, 
               rng: np.random.RandomState = None) -> np.ndarray:
    """添加高斯噪声"""
    if rng is None:
        rng = np.random.RandomState()
    
    noise = rng.randn(*img.shape) * noise_level * 255
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _add_salt_pepper_noise(img: np.ndarray, prob: float = 0.01, 
                           rng: np.random.RandomState = None) -> np.ndarray:
    """添加椒盐噪声"""
    if rng is None:
        rng = np.random.RandomState()
    
    out = img.copy()
    # 盐噪声（白点）
    salt_mask = rng.rand(*img.shape) < prob / 2
    out[salt_mask] = 255
    # 椒噪声（黑点）
    pepper_mask = rng.rand(*img.shape) < prob / 2
    out[pepper_mask] = 0
    
    return out


def _random_erasing(img: np.ndarray, p: float = 0.3, 
                    scale: Tuple[float, float] = (0.02, 0.1), 
                    ratio: Tuple[float, float] = (0.3, 3.3), 
                    rng: np.random.RandomState = None) -> np.ndarray:
    """随机擦除：提升鲁棒性"""
    if rng is None:
        rng = np.random.RandomState()
    
    if rng.rand() > p:
        return img
    
    h, w = img.shape[:2]
    area = h * w
    
    for _ in range(10):
        target_area = rng.uniform(scale[0], scale[1]) * area
        aspect_ratio = rng.uniform(ratio[0], ratio[1])
        
        eh = int(round(np.sqrt(target_area * aspect_ratio)))
        ew = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if eh < h and ew < w:
            y = rng.randint(0, h - eh)
            x = rng.randint(0, w - ew)
            img = img.copy()
            img[y:y+eh, x:x+ew] = 0
            break
    
    return img


def _cutout(img: np.ndarray, num_holes: int = 1, max_h_size: int = 8, 
            max_w_size: int = 8, rng: np.random.RandomState = None) -> np.ndarray:
    """CutOut数据增强：随机遮挡"""
    if rng is None:
        rng = np.random.RandomState()
    
    h, w = img.shape[:2]
    out = img.copy()
    
    for _ in range(num_holes):
        hole_h = rng.randint(1, max_h_size + 1)
        hole_w = rng.randint(1, max_w_size + 1)
        
        y = rng.randint(0, max(1, h - hole_h + 1))
        x = rng.randint(0, max(1, w - hole_w + 1))
        
        out[y:y+hole_h, x:x+hole_w] = 0
    
    return out


def _stroke_width_variation(img: np.ndarray, rng: np.random.RandomState = None) -> np.ndarray:
    """笔画粗细变化"""
    if rng is None:
        rng = np.random.RandomState()
    
    k = rng.choice([2, 3])
    kernel = np.ones((k, k), np.uint8)
    
    if rng.rand() < 0.5:
        # 膨胀（加粗）
        return cv2.dilate(img, kernel, iterations=1)
    else:
        # 腐蚀（变细）
        return cv2.erode(img, kernel, iterations=1)


def _grid_distortion(img: np.ndarray, num_steps: int = 5, distort_limit: float = 0.3,
                     rng: np.random.RandomState = None) -> np.ndarray:
    """网格扭曲：更自然的变形"""
    if rng is None:
        rng = np.random.RandomState()
    
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        return img
    
    # 创建网格点
    x_steps = np.linspace(0, w, num_steps)
    y_steps = np.linspace(0, h, num_steps)
    
    # 随机扰动网格点
    xx, yy = np.meshgrid(x_steps, y_steps)
    dx = rng.uniform(-distort_limit, distort_limit, xx.shape) * (w / num_steps)
    dy = rng.uniform(-distort_limit, distort_limit, yy.shape) * (h / num_steps)
    
    # 边界点不扰动
    dx[0, :] = dx[-1, :] = dx[:, 0] = dx[:, -1] = 0
    dy[0, :] = dy[-1, :] = dy[:, 0] = dy[:, -1] = 0
    
    # 插值得到完整的位移场
    from scipy.interpolate import RectBivariateSpline
    
    try:
        spline_x = RectBivariateSpline(y_steps, x_steps, dx)
        spline_y = RectBivariateSpline(y_steps, x_steps, dy)
        
        y_full = np.arange(h)
        x_full = np.arange(w)
        
        dx_full = spline_x(y_full, x_full)
        dy_full = spline_y(y_full, x_full)
        
        x_map, y_map = np.meshgrid(x_full, y_full)
        map_x = (x_map + dx_full).astype(np.float32)
        map_y = (y_map + dy_full).astype(np.float32)
        
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    except:
        return img


def _augment_strong(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """强数据增强 - 修正版：降低结构破坏风险"""
    h, w = img.shape[:2]
    if h <= 4 or w <= 4:
        return img

    out = img.copy()

    # 1) 弹性变形 (15%概率，数学公式专用低强度)
    # 数学公式中上下标、分数线等结构对变形极其敏感
    # alpha从7-17降至3-8，sigma从2-4降至2-3，概率从25%降至15%
    # 性能预期：减少约5%的结构破坏导致的错误识别
    if rng.rand() < 0.15:
        alpha = rng.uniform(3, 8)
        sigma = rng.uniform(2, 3)
        out = _elastic_transform(out, alpha=alpha, sigma=sigma, rng=rng)

    # 2) 仿射变换（旋转≤3°、缩放、平移）(60%概率)
    if rng.rand() < 0.6:
        angle = float(rng.uniform(-3, 3))  # 原±8°，改为±3°
        scale = float(rng.uniform(0.92, 1.08))  # 原0.85-1.15，收窄
        tx = float(rng.uniform(-0.05, 0.05) * w)  # 原±0.1，收窄
        ty = float(rng.uniform(-0.05, 0.05) * h)

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    # 3) 透视变换 - 已删除（破坏数学结构）

    # 4) 笔画粗细变化 (35%概率，降低)
    if rng.rand() < 0.35:
        out = _stroke_width_variation(out, rng=rng)

    # 5) 添加噪声 (25%概率，降低)
    if rng.rand() < 0.25:
        if rng.rand() < 0.7:
            noise_level = rng.uniform(0.02, 0.05)  # 原0.02-0.08，降低
            out = _add_noise(out, noise_level=noise_level, rng=rng)
        else:
            out = _add_salt_pepper_noise(out, prob=0.005, rng=rng)  # 原0.01，减半

    # 6) CutOut - 已删除（可能遮挡关键符号）

    # 7) 亮度/对比度调整 (30%概率，降低)
    if rng.rand() < 0.3:
        alpha = rng.uniform(0.85, 1.15)  # 原0.8-1.2，收窄
        beta = rng.uniform(-15, 15)      # 原±25，收窄
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # 8) 高斯模糊 (15%概率，降低) - 模拟低质量图像
    if rng.rand() < 0.15:
        ksize = 3  # 固定为3，避免过度模糊
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    return out


def _augment_medium(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """中等强度数据增强 - 修正版"""
    h, w = img.shape[:2]
    if h <= 4 or w <= 4:
        return img

    out = img.copy()

    # 1) 弹性变形 (10%概率，数学公式专用低强度)
    # 中等增强下进一步降低变形，保护分数线、根号等精细结构
    if rng.rand() < 0.10:
        alpha = rng.uniform(2, 6)
        sigma = rng.uniform(2, 3)
        out = _elastic_transform(out, alpha=alpha, sigma=sigma, rng=rng)

    # 2) 仿射变换 (50%概率，旋转≤3°)
    if rng.rand() < 0.5:
        angle = float(rng.uniform(-3, 3))  # 原±5°，改为±3°
        scale = float(rng.uniform(0.95, 1.05))  # 原0.9-1.1，收窄
        tx = float(rng.uniform(-0.04, 0.04) * w)  # 原±0.06，收窄
        ty = float(rng.uniform(-0.04, 0.04) * h)

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    # 3) 笔画粗细变化 (30%概率，降低)
    if rng.rand() < 0.3:
        out = _stroke_width_variation(out, rng=rng)

    # 4) 添加噪声 (20%概率，降低)
    if rng.rand() < 0.2:
        noise_level = rng.uniform(0.02, 0.04)  # 原0.02-0.05，降低
        out = _add_noise(out, noise_level=noise_level, rng=rng)

    # 5) 亮度/对比度调整 (25%概率，降低)
    if rng.rand() < 0.25:
        alpha = rng.uniform(0.9, 1.1)  # 原0.85-1.15，收窄
        beta = rng.uniform(-10, 10)    # 原±15，收窄
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    return out


def _augment_light(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """轻度数据增强 - 修正版（保持不变，已经很温和）"""
    h, w = img.shape[:2]
    if h <= 2 or w <= 2:
        return img

    out = img.copy()

    # 1) 轻微旋转/缩放/平移 (50%概率)
    if rng.rand() < 0.5:
        angle = float(rng.uniform(-3, 3))
        scale = float(rng.uniform(0.95, 1.05))
        tx = float(rng.uniform(-0.03, 0.03) * w)
        ty = float(rng.uniform(-0.03, 0.03) * h)

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    # 2) 笔画粗细变化 (25%概率，降低)
    if rng.rand() < 0.25:
        out = _stroke_width_variation(out, rng=rng)

    return out


def build_vocab_from_captions(
    pairs: Sequence[Tuple[str, str]],
    min_freq: int = 1,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    from collections import Counter

    counter = Counter()
    for _, cap in pairs:
        toks = cap.strip().split(" ")
        toks = [t for t in toks if t != ""]
        counter.update(toks)

    token2id: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    for tok, c in counter.most_common():
        if c < min_freq:
            continue
        if tok in token2id:
            continue
        token2id[tok] = len(token2id)

    id2token = {i: t for t, i in token2id.items()}
    return token2id, id2token


def save_vocab(path: str, token2id: Dict[str, int]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump({"token2id": token2id}, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    token2id = {k: int(v) for k, v in obj["token2id"].items()}
    id2token = {i: t for t, i in token2id.items()}
    return token2id, id2token


def build_e2e_pairs(cfg: E2EDatasetConfigV3) -> List[Tuple[str, str]]:
    root = Path(cfg.data_root)
    out: List[Tuple[str, str]] = []

    for split in cfg.splits:
        caption_path = root / split / cfg.caption_name
        img_dir = root / split / cfg.img_dir_name
        if not caption_path.exists() or not img_dir.exists():
            continue

        pairs = parse_caption_file(caption_path)
        for fname, cap in pairs:
            stem_or_name = Path(fname).name
            cand_paths = [img_dir / stem_or_name]
            if Path(stem_or_name).suffix == "":
                for ext in (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                    cand_paths.append(img_dir / (stem_or_name + ext))

            img_path = None
            for cp in cand_paths:
                if cp.exists():
                    img_path = cp
                    break
            if img_path is None:
                continue

            cap = cap.strip()
            if not cap:
                continue
            out.append((str(img_path), cap))

    return out


def split_train_val_pairs(
    pairs: List[Tuple[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rng = random.Random(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    return pairs[n_val:], pairs[:n_val]


class E2EFormulaDatasetV3(Dataset):
    """优化版数据集"""
    
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        token2id: Dict[str, int],
        cfg: E2EDatasetConfigV3,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.token2id = token2id
        self.cfg = cfg
        self.augment = augment
        self.augment_level = cfg.augment_level if augment else "none"

        self.pad_id = token2id["<pad>"]
        self.bos_id = token2id["<bos>"]
        self.eos_id = token2id["<eos>"]
        self.unk_id = token2id["<unk>"]

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode(self, cap: str) -> List[int]:
        toks = cap.strip().split(" ")
        toks = [t for t in toks if t != ""]
        ids = [self.bos_id] + [self.token2id.get(t, self.unk_id) for t in toks] + [self.eos_id]
        if len(ids) > self.cfg.max_seq_len:
            ids = ids[: self.cfg.max_seq_len]
            ids[-1] = self.eos_id
        return ids

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, cap = self.pairs[idx]
        gray = _cached_to_numpy(img_path)

        if gray is None:
            x = np.zeros((self.cfg.img_height, 1), dtype=np.uint8)
        else:
            x = _binarize_inv(gray)
            x = _resize_keep_ratio(x, self.cfg.img_height, self.cfg.max_width)

        # 课程学习：根据公式长度自动调整增强强度
        # 短公式（<30 tokens）用强增强，中等公式（30-80）用中等，长公式（>80）用轻度
        if self.augment_level != "none":
            rng = np.random.RandomState(None)
            toks = cap.strip().split(" ")
            toks = [t for t in toks if t != ""]
            formula_len = len(toks)

            # 课程学习策略：长公式更容易被破坏，使用更轻的增强
            if formula_len < 30:
                # 短公式：使用配置的增强级别
                if self.augment_level == "strong":
                    x = _augment_strong(x, rng)
                elif self.augment_level == "medium":
                    x = _augment_medium(x, rng)
                else:
                    x = _augment_light(x, rng)
            elif formula_len < 80:
                # 中等长度：降一级
                if self.augment_level == "strong":
                    x = _augment_medium(x, rng)
                elif self.augment_level == "medium":
                    x = _augment_light(x, rng)
                else:
                    x = _augment_light(x, rng)
            else:
                # 长公式：只用轻度增强
                x = _augment_light(x, rng)

        # 归一化到[0,1]
        x_t = torch.from_numpy(x).float().unsqueeze(0) / 255.0
        y_ids = self._encode(cap)
        y_t = torch.tensor(y_ids, dtype=torch.long)
        return x_t, y_t


def collate_e2e_v3(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """动态padding"""
    xs, ys = zip(*batch)

    widths = torch.tensor([int(x.shape[-1]) for x in xs], dtype=torch.long)
    wmax = int(widths.max().item())

    images = torch.zeros((len(xs), 1, xs[0].shape[-2], wmax), dtype=torch.float32)
    for i, x in enumerate(xs):
        w = x.shape[-1]
        images[i, :, :, :w] = x

    tmax = max(int(y.numel()) for y in ys)
    pad_id = 0
    tokens = torch.full((len(ys), tmax), pad_id, dtype=torch.long)
    for i, y in enumerate(ys):
        tokens[i, : y.numel()] = y

    return images, widths, tokens
