#!/usr/bin/env python3
"""根据 CHARSET 生成目标字体的"缺失字补集"（类似 HanziGen 的缺字补全）。

原理：
  1. 用 fontTools 读取目标字体的 cmap，计算  CHARSET 字符集 - 目标字体已覆盖字符 = 缺失字；
  2. 缺失字须能被参照字体（source-font）渲染（模型以参照字体的字形作为 content 输入）；
  3. 样式参考图来自目标字体自身（与训练时 ref 网格一致），字符标签用字符集内排序索引
     （与 data_processing/pipeline.py 的 charset_index 一致）；
  4. 用训练好的 LoRA checkpoint 逐个生成缺失字 PNG。

用法：
    python scripts/generate_missing_chars.py \\
        --checkpoint outputs/<字体>/checkpoint-last.pth \\
        --target-font fonts/<目标字体>.ttf \\
        --source-font fonts/<参照字体>.ttf \\
        --charset gb2312 \\
        --output-dir outputs/<字体>/missing_chars

提示：想让"字符标签"与预训练/训练阶段完全对齐，训练时建议用完整字符集
（TRAIN_CHARS_PER_FONT 覆盖整个 CHARSET，MAX_CHARS_PER_FONT 设为 None）。
"""
import argparse
import os
import random
import sys
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

# 把仓库根目录加入 sys.path，脚本可在任意位置运行
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_processing.charsets import get_charset_codepoints
from data_processing.font_utils import GlyphRenderer, get_cjk_codepoints, load_font
from data_processing.pipeline import create_combined_image, create_reference_grid, _extract_ref
# V100 专用：Cell5 在检测到 V100（cc<8，不支持 bf16 原生编译）时设置环境变量
# ZI2ZI_V100=1，这里据此切换到禁用 torch.compile 的 generate_chars_V100 版本，
# 从而消除 Inductor "does not support bfloat16 compilation natively, skipping" 刷屏警告。
if os.environ.get("ZI2ZI_V100") == "1":
    from generate_chars_V100 import DEFAULT_STEPS_BY_METHOD, patch_torch_for_device, resolve_device
else:
    from generate_chars import DEFAULT_STEPS_BY_METHOD, patch_torch_for_device, resolve_device
from util.lora_utils import _is_lora_state_dict, inject_lora
from util.misc import get_amp_dtype


def get_args_parser():
    parser = argparse.ArgumentParser("Generate Missing Glyphs (补集)", add_help=False)
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的 LoRA checkpoint")
    parser.add_argument("--target-font", type=str, required=True, help="缺字的目标字体（要补全的字体）")
    parser.add_argument("--source-font", type=str, required=True, help="参照字体（提供 content 字形，须覆盖缺失字）")
    parser.add_argument("--charset", type=str, default="gb2312", help="补全基准字符集")
    parser.add_argument("--output-dir", type=str, default="./missing_output", help="输出目录")
    parser.add_argument("--resolution", type=int, default=256, help="渲染分辨率（须与训练一致）")
    parser.add_argument("--num-images", type=int, default=None, help="最多生成多少个缺失字（默认全部）")
    parser.add_argument("--batch-size", type=int, default=64, help="推理批量")
    parser.add_argument("--cfg", type=float, default=None, help="CFG 引导强度（默认沿用 checkpoint）")
    parser.add_argument("--num-sampling-steps", type=int, default=None, help="采样步数（默认按方法）")
    parser.add_argument("--sampling-method", type=str, default=None,
                        choices=["euler", "heun", "ab2"], help="采样方法（默认沿用 checkpoint）")
    parser.add_argument("--pairwise", type=str, default="src_gen",
                        choices=["src_gen", "target_gen", "none"],
                        help="是否输出 源字形|生成结果 对比图")
    parser.add_argument("--ref-chars", type=str, default="",
                        help="逗号分隔的样式参考字（默认自动从目标字体可渲染的字里挑）")
    parser.add_argument("--ref-count", type=int, default=8, help="自动挑选的样式参考字数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser


def main(args):
    device = resolve_device(args.device)
    use_cuda_amp = device.type == "cuda"
    patch_torch_for_device(device)

    # ============ 1. 加载 checkpoint 与模型（与 generate_chars.py 完全一致） ============
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint["args"]

    from denoiser import Denoiser
    model = Denoiser(ckpt_args)
    state_dict = checkpoint.get("model_ema1", checkpoint.get("model", checkpoint))
    is_lora = _is_lora_state_dict(state_dict)
    if is_lora:
        lora_r = getattr(ckpt_args, "lora_r", 8)
        lora_alpha = getattr(ckpt_args, "lora_alpha", 16)
        lora_dropout = getattr(ckpt_args, "lora_dropout", 0.0)
        targets_str = getattr(ckpt_args, "lora_targets", "qkv,proj,w12,w3")
        targets = [t.strip() for t in targets_str.split(",") if t.strip()]
        replaced = inject_lora(model.net, targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        print(f"[补集] LoRA checkpoint: 注入 {replaced} 个模块 (r={lora_r}, alpha={lora_alpha})")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[补集] 注意: state_dict 缺失 {len(missing)} / 多余 {len(unexpected)} 项")
    model.to(device)
    model.eval()

    # 采样参数：CLI > checkpoint > 默认
    args.sampling_method = args.sampling_method or getattr(ckpt_args, "sampling_method", "heun")
    args.cfg = args.cfg if args.cfg is not None else getattr(ckpt_args, "cfg", 4.0)
    args.num_sampling_steps = args.num_sampling_steps or DEFAULT_STEPS_BY_METHOD[args.sampling_method]
    model.cfg_scale = args.cfg
    model.steps = args.num_sampling_steps
    model.method = args.sampling_method
    model.cfg_interval = (getattr(ckpt_args, "interval_min", 0.0), getattr(ckpt_args, "interval_max", 1.0))

    print(f"[补集] 采样: {args.sampling_method}, steps={args.num_sampling_steps}, cfg={args.cfg}")

    # ============ 2. 计算缺失字补集 ============
    charset_cps = get_charset_codepoints(args.charset)
    index_map = {cp: i for i, cp in enumerate(sorted(charset_cps))}  # 与 pipeline 的 charset_index 一致

    target_font, target_path = load_font(args.target_font)
    target_cps = get_cjk_codepoints(target_font)
    missing_all = sorted(charset_cps - target_cps)
    covered = len(charset_cps & target_cps)
    print(f"[补集] 字符集 {args.charset} 共 {len(charset_cps)} 字 | "
          f"目标字体覆盖 {covered} 字 | 缺失 {len(missing_all)} 字")

    source_font, source_path = load_font(args.source_font)
    source_cps = get_cjk_codepoints(source_font)
    usable = [cp for cp in missing_all if cp in source_cps]
    skipped = len(missing_all) - len(usable)
    if skipped:
        print(f"[补集] 源字体无法渲染的缺失字 {skipped} 个（已跳过，请在 missing_chars.txt 中查看）")

    num_chars = getattr(ckpt_args, "num_chars", None)
    if num_chars:
        before = len(usable)
        usable = [cp for cp in usable if index_map[cp] < num_chars]
        dropped = before - len(usable)
        if dropped:
            print(f"[补集] 超出字符嵌入空间(num_chars={num_chars})的字 {dropped} 个（已跳过）")

    if not usable:
        print("[补集] 没有可生成的缺失字：目标字体已覆盖全部字符集，或源字体无法渲染缺失字。")
        return

    if args.num_images:
        usable = usable[: args.num_images]
    n = len(usable)
    print(f"[补集] 待生成 {n} 个缺失字")

    # ============ 3. 样式参考图（来自目标字体自身，与训练 ref 网格一致） ============
    ref_pool = sorted(target_cps & charset_cps)
    if args.ref_chars:
        wanted = [ord(ch) for ch in args.ref_chars.replace("，", ",") if ch.strip() and ch != ","]
        refs = [cp for cp in wanted if cp in target_cps]
        if len(refs) < len(wanted):
            print(f"[补集] --ref-chars 中有 {len(wanted) - len(refs)} 个字目标字体无法渲染，已忽略")
    else:
        rng = random.Random(args.seed)
        refs = rng.sample(ref_pool, min(len(ref_pool), args.ref_count))
    if not refs:
        raise RuntimeError("[补集] 目标字体在字符集内没有可用字形，无法提供样式参考。")
    refs = refs[:8]
    print(f"[补集] 样式参考字: {''.join(chr(c) for c in refs)}")

    # 目标字体渲染器（分辨率与训练一致），构造 2x4 ref 网格后抽取 style 图（128x128）
    target_renderer = GlyphRenderer(str(target_path), args.resolution)
    white = Image.new("RGB", (args.resolution, args.resolution), (255, 255, 255))
    ref_grid_1 = create_reference_grid(target_renderer, refs[:4])
    ref_grid_2 = create_reference_grid(target_renderer, refs[4:]) if len(refs) > 4 else None
    if ref_grid_1 is None or (len(refs) > 4 and ref_grid_2 is None):
        raise RuntimeError("[补集] 样式参考字渲染失败。")
    combined = create_combined_image(white, white, ref_grid_1,
                                     ref_grid_2 if ref_grid_2 is not None else ref_grid_1)
    ref_img = _extract_ref(combined, 0, 128)

    # ============ 4. 渲染 content 字形（来自参照字体） ============
    src_renderer = GlyphRenderer(str(source_path), args.resolution)
    content_list = []
    for cp in usable:
        img = src_renderer.render(cp)
        if img is not None:
            content_list.append((cp, img))
    usable = [cp for cp, _ in content_list]
    n = len(usable)
    if n == 0:
        print("[补集] 参照字体无法渲染任何缺失字。")
        return

    font_labels = np.zeros(n, dtype=np.int64)
    char_labels = np.array([index_map[cp] for cp in usable], dtype=np.int64)
    unicode_labels = np.array(usable, dtype=np.int64)
    content_images = np.empty((n, 3, 256, 256), dtype=np.uint8)
    style_images = np.empty((n, 3, 128, 128), dtype=np.uint8)
    ref_arr = np.array(ref_img).transpose(2, 0, 1)
    for i, (cp, img) in enumerate(content_list):
        content_images[i] = np.array(img).transpose(2, 0, 1)
        style_images[i] = ref_arr

    # ============ 5. 推理生成 ============
    gen_folder = os.path.join(args.output_dir, "generated")
    compare_folder = os.path.join(args.output_dir, "compare") if args.pairwise != "none" else None
    os.makedirs(gen_folder, exist_ok=True)
    if compare_folder:
        os.makedirs(compare_folder, exist_ok=True)

    # 断点续传：扫描已生成的 PNG，已存在的字符直接跳过（中止后重跑只算剩余部分，省机时）
    done_cps = set()
    if os.path.isdir(gen_folder):
        for fn in os.listdir(gen_folder):
            if fn.startswith("U+") and fn.endswith(".png"):
                try:
                    done_cps.add(int(fn[2:6], 16))
                except ValueError:
                    pass
    # 兼容旧文件名格式 {font:04d}_U+{cp:04X}.png
    for fn in os.listdir(gen_folder):
        if "_U+" in fn and fn.endswith(".png"):
            try:
                done_cps.add(int(fn.split("_U+")[1][:4], 16))
            except ValueError:
                pass
    todo_idx = [i for i, cp in enumerate(usable) if cp not in done_cps]
    if todo_idx:
        print(f"[补集] 断点续传: 已存在 {n - len(todo_idx)}/{n} 个，跳过，只生成剩余 {len(todo_idx)} 个")
    else:
        print(f"[补集] 全部 {n} 个缺失字已生成，无需重算")

    for s in range(0, len(todo_idx), args.batch_size):
        idx_b = todo_idx[s:s + args.batch_size]
        end_show = s + len(idx_b)
        font_b = torch.from_numpy(font_labels[idx_b]).long().to(device)
        char_b = torch.from_numpy(char_labels[idx_b]).long().to(device)
        style_b = torch.from_numpy(style_images[idx_b].copy()).float().to(device) / 255.0 * 2.0 - 1.0
        content_b = torch.from_numpy(content_images[idx_b].copy()).float().to(device) / 255.0 * 2.0 - 1.0
        labels = (font_b, char_b, style_b, content_b)

        with (torch.amp.autocast("cuda", dtype=get_amp_dtype()) if use_cuda_amp else nullcontext()):
            generated = model.generate(labels)

        generated = (generated + 1) / 2
        generated = generated.detach().cpu()

        for j, img_id in enumerate(idx_b):
            cp = usable[img_id]
            filename = f"{int(font_labels[img_id]):04d}_U+{cp:04X}"
            gen_img = np.round(np.clip(generated[j].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]  # RGB -> BGR
            cv2.imwrite(os.path.join(gen_folder, f"{filename}.png"), gen_img)
            if compare_folder and args.pairwise == "src_gen":
                src_img = content_images[img_id].transpose([1, 2, 0])[:, :, ::-1]
                cv2.imwrite(os.path.join(compare_folder, f"{filename}.png"),
                            np.concatenate([src_img, gen_img], axis=1))
        print(f"[补集] 续跑进度 {end_show}/{len(todo_idx)}（已跳过 {n - len(todo_idx)} 个）...")

    # 缺失字清单（U+XXXX\t字符）
    with open(os.path.join(args.output_dir, "missing_chars.txt"), "w", encoding="utf-8") as f:
        f.write(f"# charset={args.charset} target={os.path.basename(args.target_font)} "
                f"total_missing={len(missing_all)} generated={n}\n")
        for cp in usable:
            f.write(f"U+{cp:04X}\t{chr(cp)}\n")

    print(f"[补集] 完成！共生成 {n} 个缺失字 -> {gen_folder}")
    print(f"[补集] 缺失字清单 -> {os.path.join(args.output_dir, 'missing_chars.txt')}")


if __name__ == "__main__":
    _args = get_args_parser().parse_args()
    main(_args)
