"""utils.py

通用工具函数 & 公共常量
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

__all__ = [
    "set_random_seed",
    "binarize",
    "resize_pad",
    "detect_contours",
]


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- 图像相关 ---------------- #

def binarize(img: np.ndarray) -> np.ndarray:
    """自适应阈值 + 取反，返回单通道二值图。"""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    return cv2.bitwise_not(binarized)


def resize_pad(img: np.ndarray, size: Tuple[int, int] = (32, 32), pad_color: int = 255) -> np.ndarray:
    """等比缩放 + 填充到固定尺寸。返回单通道 uint8。"""
    h, w = img.shape[:2]
    sh, sw = size

    if h == 0 or w == 0:
        return np.full(size, pad_color, dtype=np.uint8)

    interp = cv2.INTER_AREA if (h > sh or w > sw) else cv2.INTER_CUBIC
    aspect = w / h

    if aspect > 1:  # 宽 > 高
        new_w = sw
        new_h = int(round(new_w / aspect))
        pad_top = (sh - new_h) // 2
        pad_bot = sh - new_h - pad_top
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = int(round(new_h * aspect))
        pad_left = (sw - new_w) // 2
        pad_right = sw - new_w - pad_left
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_top = pad_bot = pad_left = pad_right = 0

    scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled = cv2.copyMakeBorder(
        scaled, pad_top, pad_bot, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=pad_color,
    )
    return scaled


# ---------------- 轮廓检测 ---------------- #

def _get_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def detect_contours(img_path: str) -> List[Tuple[int, int, int, int]]:
    """返回合并后的字符级 bounding box 列表 (x, y, w, h)。"""
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return []

    bin_img = binarize(img_gray)
    # OpenCV 的 findContours 期望前景为白色(255)，背景为黑色(0)
    # binarize() 返回的是“取反后的二值图”（笔画为白）。这里再取反会导致找不到轮廓。
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 20:
            continue
        rects.append([x, y, w, h])

    # 合并 x 方向重叠的矩形
    merged = []
    rects.sort(key=lambda r: r[0])
    while rects:
        cur_x, cur_y, cur_w, cur_h = rects.pop(0)
        to_remove = []
        for idx, (x, y, w, h) in enumerate(rects):
            if _get_overlap((cur_x, cur_x + cur_w), (x, x + w)) > 1:
                new_x = min(cur_x, x)
                new_w = max(cur_x + cur_w, x + w) - new_x
                new_y = min(cur_y, y)
                new_h = max(cur_y + cur_h, y + h) - new_y
                cur_x, cur_y, cur_w, cur_h = new_x, new_y, new_w, new_h
                to_remove.append(idx)
        for i in sorted(to_remove, reverse=True):
            rects.pop(i)
        merged.append((cur_x, cur_y, cur_w, cur_h))
    return merged
