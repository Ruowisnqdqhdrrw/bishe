"""build_charset.py

用途：
- 扫描 CROHME 数据集各子目录的 caption.txt
- 统计标注中出现的符号（token）集合与频次
- 生成 train/charset.txt（按频次降序）

运行示例：
python tools/build_charset.py --data_root "D:/CROHME/CROHME" --splits 2014 2016 2019 train

说明：
- 本脚本不会修改原始数据，只会读取 caption.txt 并在本项目内输出 charset 文件。
- token 化规则会尽量兼容常见 LaTeX/符号表示；后续如发现 caption 格式特殊，可再微调。
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path


LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")

# 常见多字符运算符/关键字（按需要可扩充）
MULTI_CHAR_TOKENS = {
    "<=", ">=", "!=", "==",
    "->", "<-", "<->",
    "...",
}

# 需要保留为单独 token 的单字符符号
SINGLE_CHAR_TOKENS = set(list("+-*/=()[]{}^_.,;:|<>!\\"))


def _normalize_caption(s: str) -> str:
    s = s.strip()
    # 统一空白
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize_caption(expr: str) -> list[str]:
    """将 caption 表达式拆成 token 序列。

    规则（偏保守）：
    1) 优先识别 LaTeX 命令，如 \int \frac \sqrt \sin
    2) 识别多字符运算符，如 <= >= !=
    3) 数字按单字符拆分（0-9），便于字符级 CNN
    4) 字母按单字符拆分（a-zA-Z），便于字符级 CNN
    5) 其它符号按单字符拆分

    注意：
    - 该 token 化用于“字符级”分类任务，会刻意把数字/字母拆成单字符。
    - 像 \int 这类符号在图像上通常是一个整体符号，保留为一个 token。
    """

    expr = _normalize_caption(expr)
    tokens: list[str] = []

    i = 0
    while i < len(expr):
        # 跳过空格
        if expr[i].isspace():
            i += 1
            continue

        # LaTeX 命令
        if expr[i] == "\\":
            m = LATEX_CMD_RE.match(expr, i)
            if m:
                tokens.append(m.group(0))
                i = m.end()
                continue
            # 单独反斜杠
            tokens.append("\\")
            i += 1
            continue

        # 多字符 token
        matched_multi = False
        for t in sorted(MULTI_CHAR_TOKENS, key=len, reverse=True):
            if expr.startswith(t, i):
                tokens.append(t)
                i += len(t)
                matched_multi = True
                break
        if matched_multi:
            continue

        ch = expr[i]

        # 数字/字母：单字符
        if ch.isdigit() or ch.isalpha():
            tokens.append(ch)
            i += 1
            continue

        # 常见符号：单字符
        if ch in SINGLE_CHAR_TOKENS:
            tokens.append(ch)
            i += 1
            continue

        # 其它不可见/未知字符也保留（便于排查）
        tokens.append(ch)
        i += 1

    return tokens


def parse_caption_file(caption_path: Path) -> list[tuple[str, str]]:
    """解析 caption.txt，返回 (filename, caption) 列表。

    兼容两类常见格式：
    1) filename<tab>caption
    2) filename caption（以空格分割，caption 可能包含空格则不适用）

    如果你的 caption.txt 格式不同，运行脚本后把前 5 行和报错贴出来，我会针对性适配。
    """

    pairs: list[tuple[str, str]] = []
    with caption_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip("\n")
            if not line.strip():
                continue
            # 优先 tab 分割
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)

            if len(parts) != 2:
                # 无法解析则跳过，但提示
                # 这里不 raise，避免整个脚本中断
                continue

            fname = parts[0].strip()
            cap = parts[1].strip()
            if fname and cap:
                pairs.append((fname, cap))

    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="CROHME 数据集根目录，如 D:/CROHME/CROHME")
    ap.add_argument("--splits", nargs="+", default=["2014", "2016", "2019", "train"], help="需要扫描的子目录")
    ap.add_argument("--out", default="train/charset.txt", help="输出 charset 文件路径")
    ap.add_argument("--show_top", type=int, default=80, help="打印 Top-N token")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"数据根目录不存在：{data_root}")

    counter: Counter[str] = Counter()
    total_lines = 0
    parsed_lines = 0

    for split in args.splits:
        caption_path = data_root / split / "caption.txt"
        if not caption_path.exists():
            print(f"[跳过] 未找到：{caption_path}")
            continue

        pairs = parse_caption_file(caption_path)
        total_lines += sum(1 for _ in caption_path.open("r", encoding="utf-8", errors="ignore"))
        parsed_lines += len(pairs)

        for _, cap in pairs:
            toks = tokenize_caption(cap)
            counter.update(toks)

        print(f"[完成] {split}: 解析 {len(pairs)} 行，累计 token 种类 {len(counter)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for tok, cnt in counter.most_common():
            f.write(f"{tok}\t{cnt}\n")

    print("\n================= 统计结果 =================")
    print(f"caption 总行数（粗略）：{total_lines}")
    print(f"成功解析行数：{parsed_lines}")
    print(f"token 总种类数：{len(counter)}")
    print(f"charset 输出：{out_path.resolve()}")

    print(f"\nTop-{args.show_top} token：")
    for tok, cnt in counter.most_common(args.show_top):
        print(f"{tok!r}\t{cnt}")

    # 额外输出：疑似 LaTeX 命令集合
    latex_cmds = sorted([t for t in counter.keys() if t.startswith("\\")])
    print(f"\nLaTeX 命令 token 数：{len(latex_cmds)}")
    if latex_cmds[:50]:
        print("示例（前 50 个）：")
        print(" ".join(latex_cmds[:50]))


if __name__ == "__main__":
    main()



