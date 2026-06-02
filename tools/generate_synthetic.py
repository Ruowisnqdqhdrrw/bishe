"""generate_synthetic.py

合成手写数学公式图片，用于扩充 CROHME 训练集。

策略：
1. 从现有 caption 中提取公式模板和子表达式
2. 通过模板组合 + 随机变异生成新公式
3. 用 matplotlib LaTeX 渲染为图片
4. 对渲染结果施加手写风格增强（弹性变形、笔画变化、噪声等）
5. 保存为 .bmp 灰度图，追加到 caption.txt

用法：
    python -m tools.generate_synthetic --num 50000
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO


# ============================================================
# 配置
# ============================================================
DATA_ROOT = Path("D:/CROHME/CROHME")
OUTPUT_SPLIT = "train"  # 合成数据存入 train split
IMG_DIR_NAME = "img"
CAPTION_NAME = "caption.txt"
SYNTHETIC_PREFIX = "syn_"  # 合成图片文件名前缀

IMG_HEIGHT = 64
MAX_WIDTH = 512


# ============================================================
# 解析现有 caption
# ============================================================
def parse_caption_file(caption_path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with caption_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)
            if len(parts) == 2:
                fname, cap = parts[0].strip(), parts[1].strip()
                if fname and cap:
                    pairs.append((fname, cap))
    return pairs


def load_all_captions() -> List[str]:
    """加载所有 split 的公式"""
    captions = []
    for split in ("train", "2014", "2016", "2019"):
        cap_path = DATA_ROOT / split / CAPTION_NAME
        if cap_path.exists():
            pairs = parse_caption_file(cap_path)
            captions.extend([cap for _, cap in pairs])
    return captions


# ============================================================
# 公式生成器
# ============================================================

# 常用变量
VARS = list("a b c d e f g h i j k l m n p q r s t u v w x y z".split())
GREEK = [r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
         r"\theta", r"\lambda", r"\mu", r"\sigma", r"\phi", r"\omega",
         r"\pi", r"\rho", r"\tau", r"\eta"]
DIGITS = list("0 1 2 3 4 5 6 7 8 9".split())
OPS = ["+ ", "- ", r"\cdot ", r"\times "]
RELATIONS = ["= ", r"\leq ", r"\geq ", r"\neq ", "< ", "> "]
FUNCS = [r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\exp"]


def _rand_var(rng: random.Random) -> str:
    if rng.random() < 0.7:
        return rng.choice(VARS)
    return rng.choice(GREEK)


def _rand_digit(rng: random.Random) -> str:
    return rng.choice(DIGITS)


def _rand_number(rng: random.Random, max_digits: int = 3) -> str:
    n = rng.randint(1, max_digits)
    return " ".join(rng.choice(DIGITS) for _ in range(n))


def _rand_simple_expr(rng: random.Random, depth: int = 0) -> str:
    """生成简单表达式"""
    if depth > 2:
        return _rand_var(rng) if rng.random() < 0.6 else _rand_number(rng, 2)

    choice = rng.random()

    if choice < 0.15:
        # 单个变量
        return _rand_var(rng)
    elif choice < 0.25:
        # 数字
        return _rand_number(rng, 2)
    elif choice < 0.45:
        # 二元运算: a + b
        a = _rand_simple_expr(rng, depth + 1)
        b = _rand_simple_expr(rng, depth + 1)
        op = rng.choice(OPS)
        return f"{a} {op}{b}"
    elif choice < 0.55:
        # 上标: x ^ { n }
        base = _rand_var(rng)
        exp = _rand_simple_expr(rng, depth + 1)
        return f"{base} ^ {{ {exp} }}"
    elif choice < 0.65:
        # 下标: x _ { i }
        base = _rand_var(rng)
        sub = rng.choice(VARS[:10] + DIGITS[:5])
        return f"{base} _ {{ {sub} }}"
    elif choice < 0.75:
        # 括号
        inner = _rand_simple_expr(rng, depth + 1)
        return f"( {inner} )"
    elif choice < 0.85:
        # 分数
        num = _rand_simple_expr(rng, depth + 1)
        den = _rand_simple_expr(rng, depth + 1)
        return rf"\frac {{ {num} }} {{ {den} }}"
    else:
        # 函数
        func = rng.choice(FUNCS)
        arg = _rand_simple_expr(rng, depth + 1)
        return f"{func} {arg}"


def gen_fraction(rng: random.Random) -> str:
    num = _rand_simple_expr(rng, 1)
    den = _rand_simple_expr(rng, 1)
    return rf"\frac {{ {num} }} {{ {den} }}"


def gen_sqrt(rng: random.Random) -> str:
    inner = _rand_simple_expr(rng, 1)
    if rng.random() < 0.3:
        n = rng.choice(["3", "4", "n"])
        return rf"\sqrt [ {n} ] {{ {inner} }}"
    return rf"\sqrt {{ {inner} }}"


def gen_sum_prod(rng: random.Random) -> str:
    op = rng.choice([r"\sum", r"\prod"])
    var = rng.choice(VARS[:5])
    lo = rng.choice(["0", "1", "2"])
    hi = rng.choice(["n", "N", r"\infty"] + DIGITS[3:8])
    body = _rand_simple_expr(rng, 1)
    return rf"{op} \limits _ {{ {var} = {lo} }} ^ {{ {hi} }} {body}"


def gen_integral(rng: random.Random) -> str:
    var = rng.choice(["x", "t", "u"])
    body = _rand_simple_expr(rng, 1)
    if rng.random() < 0.6:
        lo = rng.choice(["0", "1", "a", "- 1"])
        hi = rng.choice(["1", "n", r"\infty", "b", "2"])
        return rf"\int \limits _ {{ {lo} }} ^ {{ {hi} }} {body} d {var}"
    return rf"\int {body} d {var}"


def gen_limit(rng: random.Random) -> str:
    var = rng.choice(["x", "n", "t"])
    target = rng.choice(["0", "1", r"\infty", "a", "- 1"])
    body = _rand_simple_expr(rng, 1)
    return rf"\lim \limits _ {{ {var} \rightarrow {target} }} {body}"


def gen_matrix_small(rng: random.Random) -> str:
    """生成小矩阵 (2x2)"""
    entries = [_rand_simple_expr(rng, 2) for _ in range(4)]
    return (rf"\left ( \begin{{matrix}} {entries[0]} & {entries[1]} "
            rf"\\ {entries[2]} & {entries[3]} \end{{matrix}} \right )")


def gen_equation(rng: random.Random) -> str:
    """生成等式/不等式"""
    lhs = _rand_simple_expr(rng, 0)
    rhs = _rand_simple_expr(rng, 0)
    rel = rng.choice(RELATIONS)
    return f"{lhs} {rel}{rhs}"


def gen_subscript_superscript(rng: random.Random) -> str:
    """生成带上下标的表达式"""
    base = _rand_var(rng)
    sub = rng.choice(VARS[:5] + DIGITS[:5])
    sup = _rand_simple_expr(rng, 2)
    if rng.random() < 0.5:
        return f"{base} _ {{ {sub} }} ^ {{ {sup} }}"
    return f"{base} ^ {{ {sup} }} _ {{ {sub} }}"


def gen_trig_expr(rng: random.Random) -> str:
    """生成三角函数表达式"""
    func = rng.choice([r"\sin", r"\cos", r"\tan"])
    if rng.random() < 0.3:
        # sin^2(x)
        exp = rng.choice(["2", "3", "n"])
        arg = _rand_var(rng)
        return f"{func} ^ {{ {exp} }} {arg}"
    arg = _rand_simple_expr(rng, 1)
    return f"{func} ( {arg} )"


def gen_log_expr(rng: random.Random) -> str:
    """生成对数表达式"""
    if rng.random() < 0.4:
        base = rng.choice(["2", "1 0", "e"])
        arg = _rand_simple_expr(rng, 1)
        return rf"\log _ {{ {base} }} {arg}"
    arg = _rand_simple_expr(rng, 1)
    return rf"\ln ( {arg} )"


def gen_polynomial(rng: random.Random) -> str:
    """生成多项式"""
    var = rng.choice(["x", "t", "n"])
    n_terms = rng.randint(2, 4)
    terms = []
    for i in range(n_terms):
        coeff = _rand_number(rng, 1) if rng.random() < 0.5 else ""
        exp = n_terms - i
        if exp > 1:
            term = f"{coeff} {var} ^ {{ {exp} }}" if coeff else f"{var} ^ {{ {exp} }}"
        elif exp == 1:
            term = f"{coeff} {var}" if coeff else var
        else:
            term = _rand_number(rng, 1)
        terms.append(term)
    return " + ".join(terms)


def gen_sequence_notation(rng: random.Random) -> str:
    """生成数列记号"""
    base = rng.choice(["a", "b", "c", "u", "v"])
    sub = rng.choice(["n", "n + 1", "n - 1", "k", "i"])
    if rng.random() < 0.5:
        # 递推关系
        rhs = _rand_simple_expr(rng, 1)
        return f"{base} _ {{ {sub} }} = {rhs}"
    # 通项
    return f"{base} _ {{ {sub} }}"


def mutate_caption(caption: str, rng: random.Random) -> str:
    """对现有公式进行变异"""
    tokens = caption.split()
    if len(tokens) < 2:
        return caption

    # 随机选择变异方式
    choice = rng.random()

    if choice < 0.25 and len(tokens) > 3:
        # 替换变量名
        new_tokens = tokens.copy()
        for i, t in enumerate(new_tokens):
            if t in VARS and rng.random() < 0.4:
                new_tokens[i] = rng.choice(VARS)
        return " ".join(new_tokens)

    elif choice < 0.5 and len(tokens) > 3:
        # 替换数字
        new_tokens = tokens.copy()
        for i, t in enumerate(new_tokens):
            if t in DIGITS and rng.random() < 0.4:
                new_tokens[i] = rng.choice(DIGITS)
        return " ".join(new_tokens)

    elif choice < 0.7:
        # 替换运算符
        new_tokens = tokens.copy()
        for i, t in enumerate(new_tokens):
            if t in ["+", "-", r"\cdot", r"\times"] and rng.random() < 0.3:
                new_tokens[i] = rng.choice(["+", "-", r"\cdot", r"\times"])
        return " ".join(new_tokens)

    else:
        # 替换希腊字母
        new_tokens = tokens.copy()
        for i, t in enumerate(new_tokens):
            if t.startswith("\\") and t in GREEK and rng.random() < 0.4:
                new_tokens[i] = rng.choice(GREEK)
        return " ".join(new_tokens)


# 所有生成器及其权重
GENERATORS = [
    (gen_fraction, 12),
    (gen_sqrt, 8),
    (gen_sum_prod, 10),
    (gen_integral, 10),
    (gen_limit, 8),
    (gen_equation, 15),
    (gen_subscript_superscript, 8),
    (gen_trig_expr, 8),
    (gen_log_expr, 6),
    (gen_polynomial, 8),
    (gen_sequence_notation, 5),
]


def generate_formula(rng: random.Random, existing_captions: List[str]) -> str:
    """生成一条新公式"""
    # 40% 概率从现有公式变异，60% 概率模板生成
    if rng.random() < 0.4 and existing_captions:
        base = rng.choice(existing_captions)
        return mutate_caption(base, rng)

    # 加权随机选择生成器
    gens, weights = zip(*GENERATORS)
    gen_func = rng.choices(gens, weights=weights, k=1)[0]
    return gen_func(rng)


# ============================================================
# LaTeX 渲染
# ============================================================
def _tokens_to_latex(token_str: str) -> str:
    """将空格分隔的 token 序列转为可渲染的 LaTeX 字符串"""
    # 直接拼接，matplotlib 的 mathtext 能处理大部分
    latex = token_str.replace(" ", " ")

    # 清理多余空格
    latex = re.sub(r"\s+", " ", latex).strip()
    return latex


def render_latex_to_image(token_str: str, dpi: int = 120,
                          rng_np: np.random.RandomState = None) -> np.ndarray | None:
    """用 matplotlib 将 LaTeX 公式渲染为灰度图"""
    if rng_np is None:
        rng_np = np.random.RandomState()

    latex = _tokens_to_latex(token_str)

    # 随机选择字体大小，模拟不同手写大小
    fontsize = rng_np.randint(14, 22)

    try:
        fig = plt.figure(figsize=(8, 1.2))
        fig.patch.set_facecolor("white")

        text = fig.text(
            0.5, 0.5,
            f"${latex}$",
            fontsize=fontsize,
            ha="center", va="center",
            color="black",
        )

        # 渲染到内存
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
        plt.close(fig)

        buf.seek(0)
        img_array = np.frombuffer(buf.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None

        return img

    except Exception:
        plt.close("all")
        return None


# ============================================================
# 手写风格增强
# ============================================================
def apply_handwriting_style(img: np.ndarray, rng_np: np.random.RandomState) -> np.ndarray:
    """对渲染图施加手写风格变换"""
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return img

    # 1) 反转为前景白、背景黑（与 CROHME 二值化后一致）
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    out = bw.copy()

    # 2) 弹性变形 - 模拟手写的自然抖动
    if rng_np.rand() < 0.7:
        alpha = rng_np.uniform(2, 6)
        sigma = rng_np.uniform(2, 3)
        dx = cv2.GaussianBlur(
            (rng_np.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(
            (rng_np.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)

    # 3) 笔画粗细变化
    if rng_np.rand() < 0.5:
        k = rng_np.choice([2, 3])
        kernel = np.ones((k, k), np.uint8)
        if rng_np.rand() < 0.6:
            out = cv2.dilate(out, kernel, iterations=1)
        else:
            out = cv2.erode(out, kernel, iterations=1)

    # 4) 轻微仿射变换
    if rng_np.rand() < 0.5:
        angle = float(rng_np.uniform(-3, 3))
        scale = float(rng_np.uniform(0.95, 1.05))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    # 5) 高斯噪声
    if rng_np.rand() < 0.3:
        noise_level = rng_np.uniform(0.01, 0.04)
        noise = rng_np.randn(*out.shape) * noise_level * 255
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 6) 亮度/对比度微调
    if rng_np.rand() < 0.3:
        alpha = rng_np.uniform(0.85, 1.15)
        beta = rng_np.uniform(-10, 10)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    return out


def resize_to_target(img: np.ndarray, target_h: int = IMG_HEIGHT,
                     max_w: int = MAX_WIDTH) -> np.ndarray | None:
    """缩放到目标高度，保持宽高比，限制最大宽度"""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None

    scale = target_h / float(h)
    new_w = int(round(w * scale))
    new_w = max(1, min(new_w, max_w))

    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="合成手写数学公式图片")
    parser.add_argument("--num", type=int, default=50000, help="合成图片数量")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT), help="数据根目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dpi", type=int, default=120, help="渲染 DPI")
    parser.add_argument("--workers", type=int, default=1, help="并行数（暂不支持多进程）")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    img_dir = data_root / OUTPUT_SPLIT / IMG_DIR_NAME
    caption_path = data_root / OUTPUT_SPLIT / CAPTION_NAME

    # 确保目录存在
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rng_np = np.random.RandomState(args.seed)

    # 加载现有公式用于变异
    print("正在加载现有公式...")
    existing_captions = load_all_captions()
    print(f"已加载 {len(existing_captions)} 条现有公式")

    # 获取已有合成图片的最大编号，避免覆盖
    existing_syn = [f.stem for f in img_dir.glob(f"{SYNTHETIC_PREFIX}*.bmp")]
    start_idx = 0
    if existing_syn:
        nums = []
        for s in existing_syn:
            try:
                nums.append(int(s.replace(SYNTHETIC_PREFIX, "")))
            except ValueError:
                pass
        if nums:
            start_idx = max(nums) + 1
    print(f"合成图片起始编号: {start_idx}")

    # 打开 caption 文件追加
    cap_file = open(caption_path, "a", encoding="utf-8")

    success = 0
    fail = 0
    total = args.num

    print(f"开始合成 {total} 张图片...")
    print(f"输出目录: {img_dir}")
    print(f"Caption 文件: {caption_path}")
    print("-" * 60)

    for i in range(total):
        idx = start_idx + i

        # 生成公式
        formula = generate_formula(rng, existing_captions)

        # 渲染
        raw_img = render_latex_to_image(formula, dpi=args.dpi, rng_np=rng_np)
        if raw_img is None:
            fail += 1
            if fail % 100 == 0:
                print(f"  [警告] 累计渲染失败 {fail} 次")
            continue

        # 手写风格增强
        styled = apply_handwriting_style(raw_img, rng_np)

        # 缩放到目标尺寸
        final = resize_to_target(styled)
        if final is None:
            fail += 1
            continue

        # 检查图片是否有效（不能全黑或全白）
        if final.mean() < 1 or final.mean() > 254:
            fail += 1
            continue

        # 保存
        fname = f"{SYNTHETIC_PREFIX}{idx:06d}"
        save_path = img_dir / f"{fname}.bmp"
        cv2.imwrite(str(save_path), final)

        # 写入 caption
        cap_file.write(f"{fname}\t{formula}\n")

        success += 1

        if success % 1000 == 0:
            cap_file.flush()
            print(f"  进度: {success}/{total} 成功, {fail} 失败")

    cap_file.close()

    print("-" * 60)
    print(f"合成完成!")
    print(f"  成功: {success}")
    print(f"  失败: {fail}")
    print(f"  图片目录: {img_dir}")
    print(f"  Caption 已追加到: {caption_path}")


if __name__ == "__main__":
    main()
