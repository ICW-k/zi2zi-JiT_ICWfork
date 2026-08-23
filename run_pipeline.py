"""
zi2zi-JiT 一键流程：导入检查 -> 数据准备 -> LoRA 训练 -> 推理生成 -> 缺失字补集 -> 导出打包。

所有参数都在 config_font.py 里修改（对应 HanziGen 的 Cell 0），本文件无需修改。

用法：
    python run_pipeline.py

流程说明（参照 HanziGen 的 CloudStudio notebook 结构）：
  1. 导入：检查 fonts/ 下的目标字体、models/ 下的预训练模型
  2. 数据准备：scripts/generate_font_dataset.py 生成 train/ test/ test.npz
  3. LoRA 训练：lora_single_gpu_finetune_jit.py（AUTO_TUNE=True 时自动推算 batch_size 等）
  4. 推理生成：generate_chars.py 生成 PNG 对比图
  5. 缺失字补集：scripts/generate_missing_chars.py 按 CHARSET 补全目标字体缺失的字
  6. 导出：把生成的 PNG（可选含 LoRA checkpoint）打包成 zip 放到 exports/（可视化目录）
"""

import datetime
import glob
import os
import shutil
import subprocess
import sys
import zipfile

import torch

import config_font as CFG


def log(msg):
    print("\n[run_pipeline] " + msg)


# ===========================================================================
# 1. 导入：字体 / 预训练模型检查
# ===========================================================================
def resolve_source_font():
    """源字体（content 参照字体）：显式指定则用之，否则在 fonts/ 里自动挑一个
    不属于 TARGET_FONTS 的字体。"""
    if CFG.SOURCE_FONT:
        p = CFG.SOURCE_FONT if os.path.isabs(CFG.SOURCE_FONT) else os.path.join(CFG.PROJECT_ROOT, CFG.SOURCE_FONT)
        if not os.path.exists(p):
            raise FileNotFoundError("源字体不存在: {}".format(p))
        return p
    target_basenames = {os.path.basename(t) for t in CFG.TARGET_FONTS}
    fonts = []
    for ext in ("*.ttf", "*.otf", "*.ttc"):
        fonts += glob.glob(os.path.join(CFG.FONTS_DIR, ext))
    candidates = [f for f in fonts if os.path.basename(f) not in target_basenames]
    if not candidates:
        raise FileNotFoundError(
            "未找到源字体：请在 fonts/ 目录放一个“参照字体”（与目标字体不同），"
            "或在 config_font.py 中显式指定 SOURCE_FONT。"
        )
    return sorted(candidates)[0]


def check_imports():
    log("检查导入素材（可视化目录: fonts/ models/）...")
    os.makedirs(CFG.FONTS_DIR, exist_ok=True)
    os.makedirs(CFG.MODELS_DIR, exist_ok=True)

    for t in CFG.TARGET_FONTS:
        p = t if os.path.isabs(t) else os.path.join(CFG.PROJECT_ROOT, t)
        if not os.path.exists(p):
            raise FileNotFoundError(
                "目标字体不存在: {}\n请把字体文件上传到 {} 目录。".format(p, CFG.FONTS_DIR)
            )
    src = resolve_source_font()
    log("源字体: {}".format(src))

    ckpt_path = CFG.BASE_CHECKPOINT
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(CFG.PROJECT_ROOT, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "预训练模型不存在: {}\n请按 README 中的下载链接获取 zi2zi-JiT-B-16.pth"
            " 并放到 models/ 目录。".format(ckpt_path)
        )
    log("预训练模型: {}".format(ckpt_path))
    return src, ckpt_path


# ===========================================================================
# 2. 数据准备
# ===========================================================================
def step_data_prep():
    if not CFG.DO_DATA_PREP:
        log("跳过数据准备（DO_DATA_PREP=False）")
        return
    if os.path.exists(CFG.TRAIN_DIR) and os.path.exists(CFG.TEST_NPZ_PATH):
        log("数据集已存在，跳过生成: {}".format(CFG.DATASET_DIR))
        return

    # 只把 TARGET_FONTS 指定的字体放进一个临时目录，避免 fonts/ 下其他字体也被训练
    staging = os.path.join(CFG.DATA_DIR, "_staging_fonts")
    os.makedirs(staging, exist_ok=True)
    for t in CFG.TARGET_FONTS:
        p = t if os.path.isabs(t) else os.path.join(CFG.PROJECT_ROOT, t)
        shutil.copy2(p, staging)

    script = os.path.join(CFG.PROJECT_ROOT, "scripts", "generate_font_dataset.py")
    cmd = [
        sys.executable, script,
        "--source-font", resolve_source_font(),
        "--font-dir", staging,
        "--output-dir", CFG.DATASET_DIR,
        "--train-chars-per-font", str(CFG.TRAIN_CHARS_PER_FONT),
        "--test-chars-per-font", str(CFG.TEST_CHARS_PER_FONT),
        "--resolution", str(CFG.RESOLUTION),
        "--charset", CFG.CHARSET,
        "--train-seed", str(CFG.TRAIN_SEED),
        "--test-seed", str(CFG.TEST_SEED),
        "--num-workers", str(CFG.NUM_WORKERS_DATA_PREP),
    ]
    log("数据准备命令: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ===========================================================================
# 3. LoRA 训练（含硬件自动调优）
# ===========================================================================
def build_probe_model(args, device):
    """与 lora_single_gpu_finetune_jit.main 完全一致地构建“已注入 LoRA”的模型，
    用于 probe 实测单样本显存。"""
    torch._dynamo.config.cache_size_limit = 128
    from denoiser import Denoiser
    from util.lora_utils import (
        inject_lora, mark_only_lora_as_trainable, _is_lora_state_dict, resolve_checkpoint_path,
    )

    model = Denoiser(args)
    model.update_ema = lambda: None

    ckpt_path = resolve_checkpoint_path(args.base_checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    is_lora = _is_lora_state_dict(state_dict)
    del checkpoint

    if not is_lora:
        model.load_state_dict(state_dict, strict=True)

    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    inject_lora(model.net, targets, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    if is_lora:
        model.load_state_dict(state_dict, strict=True)

    mark_only_lora_as_trainable(model, train_font_emb=True)
    model.to(device)
    return model


def auto_tune_before_train(args):
    """根据 config 里的 AUTO_TUNE 设置推算 batch_size / gen_bsz / num_workers。"""
    from util.auto_tune import auto_tune

    device = torch.device(args.device)
    common = dict(
        model_name=CFG.MODEL,
        img_size=CFG.IMG_SIZE,
        reserve=CFG.TUNE_RESERVE,
        max_batch=CFG.TUNE_MAX_BATCH,
        max_gen_bsz=CFG.TUNE_MAX_GEN_BSZ,
        num_workers_cap=CFG.TUNE_NUM_WORKERS_CAP,
        batch_size_fallback=CFG.BATCH_SIZE,
    )
    if CFG.TUNE_METHOD == "probe" and device.type == "cuda":
        log("构建探测模型（与真实训练完全一致，含 LoRA 注入）...")
        probe_model = build_probe_model(args, device)
        tune = auto_tune(
            model=probe_model, device=device, method="probe",
            num_fonts=CFG.NUM_FONTS, num_chars=CFG.NUM_CHARS, **common,
        )
        del probe_model
        torch.cuda.empty_cache()
        return tune
    log("使用静态标定表估算 batch_size（method=table）...")
    return auto_tune(method="table", **common)


def step_train():
    if not CFG.DO_TRAIN:
        log("跳过训练（DO_TRAIN=False）")
        return

    from lora_single_gpu_finetune_jit import get_args_parser, main as lora_main

    argv = [
        "--data_path", CFG.TRAIN_DIR,
        "--test_npz_path", CFG.TEST_NPZ_PATH,
        "--output_dir", CFG.OUTPUT_DIR,
        "--base_checkpoint", CFG.BASE_CHECKPOINT,
        "--model", CFG.MODEL,
        "--img_size", str(CFG.IMG_SIZE),
        "--num_fonts", str(CFG.NUM_FONTS),
        "--num_chars", str(CFG.NUM_CHARS),
        "--lora_r", str(CFG.LORA_R),
        "--lora_alpha", str(CFG.LORA_ALPHA),
        "--lora_targets", CFG.LORA_TARGETS,
        "--lora_dropout", str(CFG.LORA_DROPOUT),
        "--proj_dropout", str(CFG.PROJ_DROPOUT),
        "--epochs", str(CFG.EPOCHS),
        "--batch_size", str(CFG.BATCH_SIZE),
        "--blr", str(CFG.BLR),
        "--min_lr", str(CFG.MIN_LR),
        "--warmup_epochs", str(CFG.WARMUP_EPOCHS),
        "--save_last_freq", str(CFG.SAVE_LAST_FREQ),
        "--P_mean", str(CFG.P_MEAN),
        "--P_std", str(CFG.P_STD),
        "--noise_scale", str(CFG.NOISE_SCALE),
        "--cfg", str(CFG.CFG),
        "--num_images", str(CFG.NUM_IMAGES),
        "--seed", str(CFG.SEED),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    if CFG.MAX_CHARS_PER_FONT is not None:
        argv += ["--max_chars_per_font", str(CFG.MAX_CHARS_PER_FONT)]

    args = get_args_parser().parse_args(argv)
    if CFG.ONLINE_EVAL:
        args.online_eval = True
    if CFG.EVAL_STEP_FOLDERS:
        args.eval_step_folders = True

    if CFG.AUTO_TUNE:
        tune = auto_tune_before_train(args)
        args.batch_size = tune["batch_size"]
        args.gen_bsz = tune["gen_bsz"]
        args.num_workers = tune["num_workers"]
    else:
        log("AUTO_TUNE=False，使用 config_font.py 中固定的 BATCH_SIZE/GEN_BSZ/NUM_WORKERS")

    log("开始 LoRA 训练：batch_size={}, gen_bsz={}, num_workers={}".format(
        args.batch_size, args.gen_bsz, args.num_workers))
    lora_main(args)


# ===========================================================================
# 4. 推理生成
# ===========================================================================
def step_generate():
    if not CFG.DO_GENERATE:
        log("跳过推理生成（DO_GENERATE=False）")
        return

    ckpt = os.path.join(CFG.OUTPUT_DIR, "checkpoint-last.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            "未找到训练好的 checkpoint: {}\n请先完成训练（DO_TRAIN=True）。".format(ckpt)
        )

    from generate_chars import get_args_parser, main as gen_main

    argv = [
        "--checkpoint", ckpt,
        "--test_npz", CFG.TEST_NPZ_PATH,
        "--output_dir", CFG.GEN_OUTPUT_DIR,
        "--device", "auto",
    ]
    args = get_args_parser().parse_args(argv)

    if CFG.GENERATE_NUM_IMAGES is not None:
        args.num_images = CFG.GENERATE_NUM_IMAGES
    if CFG.GENERATE_BATCH_SIZE is not None:
        args.batch_size = CFG.GENERATE_BATCH_SIZE
    if CFG.GENERATE_CFG is not None:
        args.cfg = CFG.GENERATE_CFG
    if CFG.GENERATE_SAMPLING_METHOD is not None:
        args.sampling_method = CFG.GENERATE_SAMPLING_METHOD
    if CFG.GENERATE_NUM_SAMPLING_STEPS is not None:
        args.num_sampling_steps = CFG.GENERATE_NUM_SAMPLING_STEPS
    if CFG.GENERATE_PAIRWISE is not None:
        args.pairwise = CFG.GENERATE_PAIRWISE

    if CFG.AUTO_TUNE:
        # 推理无反向，但 CFG 双倍前向，用 table 模式给一个保守的推理批量
        from util.auto_tune import auto_tune
        tune = auto_tune(
            method="table", model_name=CFG.MODEL, img_size=CFG.IMG_SIZE,
            reserve=CFG.TUNE_RESERVE, max_batch=CFG.TUNE_MAX_BATCH,
            max_gen_bsz=CFG.TUNE_MAX_GEN_BSZ, batch_size_fallback=CFG.GENERATE_BATCH_SIZE,
            verbose=False,
        )
        args.batch_size = tune["gen_bsz"]

    log("开始推理生成：batch_size={}".format(args.batch_size))
    gen_main(args)


# ===========================================================================
# 5. 缺失字补集：按 CHARSET 计算目标字体缺失的字并用模型补全
# ===========================================================================
def step_missing_gen():
    if not CFG.DO_MISSING_GEN:
        log("跳过缺失字补集生成（DO_MISSING_GEN=False）")
        return
    ckpt = os.path.join(CFG.OUTPUT_DIR, "checkpoint-last.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            "未找到训练好的 checkpoint: {}\n请先完成训练（DO_TRAIN=True）。".format(ckpt)
        )

    script = os.path.join(CFG.PROJECT_ROOT, "scripts", "generate_missing_chars.py")
    target = CFG.TARGET_FONTS[0]
    target_path = target if os.path.isabs(target) else os.path.join(CFG.PROJECT_ROOT, target)

    cmd = [
        sys.executable, script,
        "--checkpoint", ckpt,
        "--target-font", target_path,
        "--source-font", resolve_source_font(),
        "--charset", CFG.MISSING_CHARSET or CFG.CHARSET,
        "--output-dir", CFG.MISSING_OUTPUT_DIR,
        "--batch-size", str(CFG.MISSING_BATCH_SIZE or 32),
        "--pairwise", CFG.MISSING_PAIRWISE or "none",
        "--seed", str(CFG.SEED),
        "--device", "auto",
    ]
    if CFG.MISSING_NUM_IMAGES is not None:
        cmd += ["--num-images", str(CFG.MISSING_NUM_IMAGES)]
    if CFG.MISSING_CFG is not None:
        cmd += ["--cfg", str(CFG.MISSING_CFG)]
    if CFG.MISSING_SAMPLING_METHOD is not None:
        cmd += ["--sampling-method", str(CFG.MISSING_SAMPLING_METHOD)]
    if CFG.MISSING_NUM_SAMPLING_STEPS is not None:
        cmd += ["--num-sampling-steps", str(CFG.MISSING_NUM_SAMPLING_STEPS)]
    if CFG.MISSING_REF_CHARS:
        cmd += ["--ref-chars", CFG.MISSING_REF_CHARS]

    log("开始缺失字补集生成: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ===========================================================================
# 6. 导出：PNG 打包（+ 可选 LoRA checkpoint）
# ===========================================================================
def step_export():
    if not CFG.DO_EXPORT:
        log("跳过导出（DO_EXPORT=False）")
        return
    if not os.path.isdir(CFG.GEN_OUTPUT_DIR):
        log("未找到生成目录，跳过导出: {}".format(CFG.GEN_OUTPUT_DIR))
        return

    os.makedirs(CFG.EXPORTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = "{}_{}_{}.zip".format(CFG.EXPORT_PREFIX, CFG.FONT_TAG, stamp)
    zip_path = os.path.join(CFG.EXPORTS_DIR, zip_name)

    n_png = 0
    export_roots = [CFG.GEN_OUTPUT_DIR]
    if CFG.DO_MISSING_GEN and os.path.isdir(CFG.MISSING_OUTPUT_DIR):
        export_roots.append(CFG.MISSING_OUTPUT_DIR)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_dir in export_roots:
            for root, _, files in os.walk(root_dir):
                for f in sorted(files):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, CFG.PROJECT_ROOT)
                        zf.write(full, rel)
                        n_png += 1
        if CFG.EXPORT_INCLUDE_CHECKPOINT:
            ckpt = os.path.join(CFG.OUTPUT_DIR, "checkpoint-last.pth")
            if os.path.exists(ckpt):
                zf.write(ckpt, os.path.join("checkpoint", "checkpoint-last.pth"))

    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    log("导出完成: {}（{} 张图片，{:.1f} MB）".format(zip_path, n_png, size_mb))


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    CFG.print_summary()
    check_imports()
    step_data_prep()
    step_train()
    step_generate()
    step_missing_gen()
    step_export()
    log("全部完成。生成结果目录: {}".format(CFG.GEN_OUTPUT_DIR))
    if CFG.DO_MISSING_GEN and os.path.isdir(CFG.MISSING_OUTPUT_DIR):
        log("缺失字补集目录: {}".format(CFG.MISSING_OUTPUT_DIR))
    log("导出 zip 目录: {}".format(CFG.EXPORTS_DIR))


if __name__ == "__main__":
    main()
