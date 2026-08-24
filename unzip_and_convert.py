#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查找 Cell 6 导出的 zip → 提取补字 PNG → 自动调用 png_to_svg 转 SVG。

zip 查找顺序（取第一个存在 zip 的目录中最新一个）：
    1. <项目根>/exports/    旧版 Cell 6 输出目录
    2. <项目根>/            根目录（手动拷贝/下载到项目根的情况）
    3. <项目根>/outputs/    修改后 Cell 6 的新输出目录

提取规则（兼容 Cell 0 的三档打包开关 EXPORT_PACK_MISSING/GENERATED/CHECKPOINT，
无论打包范围如何都能取到补字图，不会误判"找不到 zip"）：
    优先提取 zip 内 missing_chars/generated/ 下的 PNG（补字图）；
    若 zip 里不含 missing_chars（即只打了 generated_chars / checkpoint），
    则回退提取 zip 内任意 .../generated/ 下的 PNG，并在日志里给出提示；
    若连 generated 也没有（只打了 checkpoint），报错退出。

流程：
    解压补字 PNG → <项目根>/generated/
    → 调用 scripts/png_to_svg.py 转成 <项目根>/SVG/
    → 转换完成后删除临时解压出的 generated/
    → SVG 目录不存在时自动创建

运行方式（脚本在项目根目录时）：
    python unzip_and_convert.py
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile


def project_root():
    """脚本位于项目根目录，直接返回脚本所在目录。"""
    return os.path.dirname(os.path.abspath(__file__))


def find_latest_zip(root):
    """按 exports/ → 根目录 → outputs/ 顺序，返回第一个有 zip 的目录里最新的 zip。"""
    candidates = [
        os.path.join(root, "exports"),
        root,
        os.path.join(root, "outputs"),
    ]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        zips = [p for p in glob.glob(os.path.join(d, "*.zip")) if zipfile.is_zipfile(p)]
        if not zips:
            continue
        zips.sort(key=os.path.getmtime, reverse=True)
        return zips[0], d
    return None, None


def select_png_entries(names):
    """从 zip 内文件名列表里挑出要解压的补字 PNG 条目，返回 (entries, source_tag)。

    source_tag 用于日志说明取自哪一类：missing / generated_fallback / none。
    """
    norm = [n.replace("\\", "/") for n in names]
    missing = [n for n in norm
               if "/missing_chars/generated/" in n and n.lower().endswith(".png")]
    if missing:
        return missing, "missing"

    # 回退：用户在 Cell 0 只打 generated_chars 时，也尽量取到 PNG（这类图同样
    # 是 U+XXXX 命名，可拼字体）。注意要排除 compare/ 下的并排对比图。
    any_gen = [n for n in norm
               if "/generated/" in n and "/compare/" not in n
               and n.lower().endswith(".png")]
    if any_gen:
        return any_gen, "generated_fallback"
    return [], "none"


def extract_entries(zip_path, entries, dest_dir):
    """把 zip 内指定 entries 平铺解压到 dest_dir（只保留文件名）。"""
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as z:
        for entry in entries:
            base = os.path.basename(entry)
            if not base:
                continue
            with z.open(entry) as src, open(os.path.join(dest_dir, base), "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def main():
    root = project_root()
    svg_dir = os.path.join(root, "SVG")
    gen_dir = os.path.join(root, "generated")

    # 1. 找 zip
    zip_path, found_in = find_latest_zip(root)
    if not zip_path:
        print("[unzip2svg] 未找到 zip（exports/、根目录、outputs/ 都没有）",
              file=sys.stderr)
        return 1
    print("[unzip2svg] 找到 zip: %s（位置: %s）" % (zip_path, found_in))

    # 2. 看 zip 里有什么，挑出要解压的补字 PNG
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    entries, source = select_png_entries(names)
    if not entries:
        # 通常是用户在 Cell 0 只开了 EXPORT_PACK_CHECKPOINT，没打任何 PNG
        has_ckpt = any(n.replace("\\", "/").lower().startswith("checkpoint/")
                       for n in names)
        if has_ckpt:
            print("[unzip2svg] zip 里只有 checkpoint 权重、没有 PNG。"
                  "请回 Cell 0 把 EXPORT_PACK_MISSING 设为 True 再重新导出。",
                  file=sys.stderr)
        else:
            print("[unzip2svg] zip 里没有可解压的 PNG（也没找到 checkpoint）",
                  file=sys.stderr)
        return 1
    if source == "missing":
        print("[unzip2svg] zip 含 missing_chars/generated/，提取补字图")
    else:
        print("[unzip2svg] zip 不含 missing_chars（可能 EXPORT_PACK_MISSING=False），"
              "回退提取 generated/ 下的 PNG")

    # 3. 清理并解压到 generated/
    if os.path.isdir(gen_dir):
        shutil.rmtree(gen_dir)
        print("[unzip2svg] 已清理旧的 generated/ 目录")
    n = extract_entries(zip_path, entries, gen_dir)
    print("[unzip2svg] 已解压 %d 张 PNG -> %s" % (n, gen_dir))

    # 4. 保证 SVG 目录存在（转换脚本也会建，这里双保险）
    os.makedirs(svg_dir, exist_ok=True)

    # 5. 调用 png_to_svg.py
    converter = os.path.join(root, "scripts", "png_to_svg.py")
    if not os.path.exists(converter):
        print("[unzip2svg] 找不到转换脚本: %s" % converter, file=sys.stderr)
        shutil.rmtree(gen_dir, ignore_errors=True)
        return 1
    print("[unzip2svg] 调用: %s --input %s --output %s"
          % (converter, gen_dir, svg_dir))
    try:
        subprocess.run([sys.executable, converter,
                        "--input", gen_dir, "--output", svg_dir], check=True)
    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)
        print("[unzip2svg] 已删除临时解压目录 %s" % gen_dir)

    n_svg = len(glob.glob(os.path.join(svg_dir, "*.svg")))
    print("[unzip2svg] 完成：SVG 输出到 %s（共 %d 个）" % (svg_dir, n_svg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
