#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量把 PNG 位图字形转换为 SVG 矢量图（供 FontForge 拼字体用）。

用法：
    python scripts/png_to_svg.py [--input <PNG目录>] [--output <SVG目录>]

默认输入目录：<项目根>/generated   默认输出目录：<项目根>/SVG
SVG 输出与 PNG 同名（扩展名换成 .svg）；SVG 目录不存在时自动创建。

转换原理（引擎按优先级自动选择）：
  - potracer Python 库（import potrace，pip install potracer，与 SVG_CONVERT
    参考项目一致，曲线质量最佳、无需系统命令）；
  - potrace 命令行（在 PATH 中时）；
  - OpenCV 轮廓 + 多边形逼近 + Chaikin 平滑（兜底）。

孔洞处理（关键）：
  SVG 的 fill-rule="evenodd" 只对同一个 <path> 元素内的多个子路径生效。
  因此所有外轮廓与其内部孔洞必须合并成单个 <path> 的多段子路径
  （如 "M..L..Z M..L..Z"），否则孔洞会被当作独立实心图形填黑，
  导致 SVG 与 PNG 不一致。本脚本三条引擎路径均按此规则输出。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import cv2
from PIL import Image


def project_root():
    """脚本位于 scripts/ 下，返回项目根目录。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_mask(img, threshold=128):
    """返回字形前景掩码（True=字形像素）。

    支持两种输入：
      - 白底黑字 RGB（本项目的补字图）：亮度 < threshold 视为笔画；
      - 透明底图：alpha >= 128 视为字形。
    """
    rgba = img.convert("RGBA")
    alpha = np.asarray(rgba)[..., 3]
    if (alpha < 255).any():
        return alpha >= 128
    gray = np.asarray(img.convert("L"))
    return gray < threshold


def _chaikin(pts, iters=1):
    """Chaikin 角点切割平滑（对折线做 1~2 次，去锯齿感）。"""
    pts = [np.asarray(p, dtype=float) for p in pts]
    for _ in range(iters):
        out = []
        n = len(pts)
        for i in range(n):
            p0, p1 = pts[i], pts[(i + 1) % n]
            out.append(0.75 * p0 + 0.25 * p1)
            out.append(0.25 * p0 + 0.75 * p1)
        pts = out
    return pts


def trace_cv2(mask, epsilon=1.0, smooth=1, min_area=6.0):
    """OpenCV 轮廓追踪：所有子路径（外轮廓+孔洞）合并进单个 <path> 的 d 字符串。

    必须用 RETR_CCOMP 区分外轮廓与孔洞：
      - 外轮廓面积 < min_area 且没有孔洞 → 视为噪声丢弃；
      - 孔洞无论多小都保留，否则字内空白会被填黑。
    返回单元素列表（合并后的 d）；无字形时返回空列表。
    """
    fg = (mask.astype(np.uint8)) * 255
    contours, hierarchy = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # (n,4): [next, prev, first_child, parent]
    subpaths = []
    for i, c in enumerate(contours):
        is_hole = int(hierarchy[i][3]) != -1
        if not is_hole and cv2.contourArea(c) < min_area:
            continue  # 独立噪声点，丢弃
        poly = cv2.approxPolyDP(c, epsilon, True)
        if len(poly) < 3:
            continue
        pts = poly.reshape(-1, 2).astype(float)
        if smooth > 0:
            pts = _chaikin(pts, smooth)
        d = "M %.1f %.1f " % (pts[0][0], pts[0][1])
        d += "L " + " ".join("%.1f %.1f" % (x, y) for x, y in pts[1:]) + " Z"
        subpaths.append(d)
    if not subpaths:
        return []
    # 合并为单个 path 元素：evenodd 规则跨子路径生效 → 孔洞正确挖空
    return [" ".join(subpaths)]


def potrace_lib_available():
    """检查是否安装了 potracer Python 库（import potrace）。"""
    try:
        import potrace  # noqa: F401
        return True
    except ImportError:
        return False


def trace_potrace_lib(mask, turdsize=2, alphamax=1.0, opttolerance=0.2,
                      blacklevel=0.5):
    """用 potracer Python 库追踪（与 SVG_CONVERT 参考项目同款引擎）。

    返回单元素列表（合并后的 d，evenodd 保留孔洞）；库不可用或追踪失败返回 None。
    """
    try:
        from potrace import POTRACE_TURNPOLICY_MINORITY, Bitmap
    except ImportError:
        return None
    try:
        fg_img = Image.fromarray((~mask).astype(np.uint8) * 255).convert("L")
        bm = Bitmap(fg_img, blacklevel=blacklevel)
        plist = bm.trace(
            turdsize=turdsize,
            turnpolicy=POTRACE_TURNPOLICY_MINORITY,
            alphamax=alphamax,
            opticurve=True,
            opttolerance=opttolerance,
        )
    except Exception:
        return None
    if not plist:
        return None
    # 所有路径合并进单个 d，配合 <g fill-rule="evenodd"> 保留孔洞
    fmt = lambda v: "%g" % v
    subpaths = []
    for curve in plist:
        s = curve.start_point
        segs = ["M%s,%s" % (fmt(s.x), fmt(s.y))]
        for seg in curve.segments:
            if seg.is_corner:
                segs.append("L%s,%sL%s,%s"
                            % (fmt(seg.c.x), fmt(seg.c.y),
                               fmt(seg.end_point.x), fmt(seg.end_point.y)))
            else:
                segs.append("C%s,%s %s,%s %s,%s"
                            % (fmt(seg.c1.x), fmt(seg.c1.y),
                               fmt(seg.c2.x), fmt(seg.c2.y),
                               fmt(seg.end_point.x), fmt(seg.end_point.y)))
        segs.append("z")
        subpaths.append("".join(segs))
    return [" ".join(subpaths)]


def trace_potrace(mask, turdsize=2):
    """用 potrace 命令行追踪（黑=字形），返回 SVG path 的 d 字符串列表；失败返回 None。"""
    # 字形画成黑色，背景白色（potrace 默认追踪黑色前景）
    fg_img = Image.fromarray((~mask).astype(np.uint8) * 255).convert("L")
    tmp_bmp = None
    try:
        fd, tmp_bmp = tempfile.mkstemp(suffix=".bmp")
        os.close(fd)
        fg_img.save(tmp_bmp, "BMP")
        res = subprocess.run(
            ["potrace", "-s", "--turdsize", str(turdsize), "-o", "-", tmp_bmp],
            capture_output=True, check=True)
        svg = res.stdout.decode("utf-8", "replace")
        return re.findall(r'\bd="([^"]*)"', svg)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    finally:
        if tmp_bmp and os.path.exists(tmp_bmp):
            os.remove(tmp_bmp)


def write_svg(path, path_data, w, h):
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<svg xmlns="http://www.w3.org/2000/svg" '
                'xmlns:xlink="http://www.w3.org/1999/xlink" '
                'viewBox="0 0 %d %d">\n' % (w, h))
        f.write('<g fill="#000000" fill-rule="evenodd" stroke="none">\n')
        for d in path_data:
            f.write('  <path d="%s"/>\n' % d)
        f.write('</g>\n</svg>\n')


def convert_one(img, threshold, epsilon, smooth, min_area, use_lib, use_cli):
    mask = build_mask(img, threshold)
    # 引擎优先级：potracer 库 → potrace 命令行 → OpenCV 兜底
    if use_lib:
        parts = trace_potrace_lib(mask, blacklevel=threshold / 255.0)
        if parts is not None:
            return parts
    if use_cli:
        parts = trace_potrace(mask)
        if parts is not None:
            return parts
    return trace_cv2(mask, epsilon, smooth, min_area)


def main():
    ap = argparse.ArgumentParser(description="PNG 位图字形 → SVG（供 FontForge 使用）")
    ap.add_argument("--input", default=None, help="PNG 输入目录（默认 <项目根>/generated）")
    ap.add_argument("--output", default=None, help="SVG 输出目录（默认 <项目根>/SVG）")
    ap.add_argument("--threshold", type=int, default=128,
                    help="二值化阈值 0-255（默认 128；越大笔画越粗）")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="多边形逼近容差，像素（默认 1.0；越小越精细）")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Chaikin 平滑迭代次数（默认 1；0=不平滑）")
    ap.add_argument("--min-area", type=float, default=6.0,
                    help="忽略面积小于该像素值的噪声轮廓")
    ap.add_argument("--no-potrace", action="store_true", help="强制不用 potrace")
    args = ap.parse_args()

    root = project_root()
    in_dir = args.input or os.path.join(root, "generated")
    out_dir = args.output or os.path.join(root, "SVG")
    os.makedirs(out_dir, exist_ok=True)

    use_lib = (not args.no_potrace) and potrace_lib_available()
    use_cli = (not args.no_potrace) and shutil.which("potrace") is not None
    if use_lib:
        print("[png2svg] 使用 potracer Python 库（曲线质量最佳）")
    elif use_cli:
        print("[png2svg] 使用 potrace 命令行（未安装 potracer 库）")
    else:
        print("[png2svg] 未找到 potrace，使用 OpenCV 轮廓追踪")

    if not os.path.isdir(in_dir):
        print("[png2svg] 输入目录不存在: %s" % in_dir, file=sys.stderr)
        return 1
    names = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".png"))
    if not names:
        print("[png2svg] 输入目录里没有 PNG: %s" % in_dir, file=sys.stderr)
        return 1

    ok = fail = 0
    for name in names:
        src = os.path.join(in_dir, name)
        dst = os.path.join(out_dir, os.path.splitext(name)[0] + ".svg")
        try:
            img = Image.open(src)
            parts = convert_one(img, args.threshold, args.epsilon,
                                args.smooth, args.min_area, use_lib, use_cli)
            if not parts:
                print("[png2svg] 跳过（未追踪到字形）: %s" % name)
                fail += 1
                continue
            write_svg(dst, parts, img.width, img.height)
            ok += 1
        except Exception as e:
            print("[png2svg] 失败 %s: %s" % (name, e))
            fail += 1

    print("[png2svg] 完成：%d 成功，%d 失败 -> %s" % (ok, fail, out_dir))
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
