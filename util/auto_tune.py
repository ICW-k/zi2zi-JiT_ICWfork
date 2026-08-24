"""自动硬件调优：根据 GPU 显存 / CPU 核数安全推算 BATCH_SIZE、NUM_WORKERS、GEN_BSZ。

为什么要自动调优？
    训练不同的字体、不同的模型变体（JiT-B/16 vs JiT-L/16）、不同的 img_size 时，
    显存占用主要由“模型前向+反向产生的激活值”决定，而这与 batch_size 近似线性相关。
    LoRA 的 r/alpha 只会改变可训练参数（优化器状态），对单样本显存的影响是二阶小量。
    因此完全可以在运行前实测“每个样本的显存增量”，再反推出安全的 batch_size，
    而不必每次凭人工经验去试。

两种调优方法：
- probe（默认，推荐）：
    在真实模型（已注入 LoRA）上以 batch=1 和 batch=4 各执行一次训练式的前向+反向，
    用 torch.cuda.max_memory_allocated() 实测单样本显存增量，再反推安全 batch。
    任何参数（模型变体 / img_size / LoRA r / alpha / encoder 配置）的变化都会被自动捕获，
    比静态表格更准。
- table（更快，但为近似值）：
    使用预标定的“单样本显存常数”表（README: JiT-B/16 @256 ≈ 0.25 GB/样本）
    加上固定基座开销来估算，不需要真正构建模型。适合 GPU 显存小、想快速出结果的情况。

典型用法：
    from util.auto_tune import auto_tune
    tune = auto_tune(method="table", model_name="JiT-B/16", img_size=256, verbose=True)
    print(tune["batch_size"], tune["num_workers"], tune["gen_bsz"])
"""

import os
import math

import torch

from util.misc import get_amp_dtype


# ---------------------------------------------------------------------------
# 标定常数（table 模式用）
# ---------------------------------------------------------------------------
# 单样本训练显存成本 (GB/sample)。来源：
#   README: JiT-B/16 @ 256px，batch_size=16 约需 4 GB -> 4/16 = 0.25 GB/样本
#   JiT-L/16 参数量约为 B 的 2 倍，激活值也近似翻倍（未实测，属于保守估算）
PER_SAMPLE_GB_TABLE = {
    ("JiT-B/16", 256): 0.25,
    ("JiT-B/16", 224): 0.20,
    ("JiT-L/16", 256): 0.50,
    ("JiT-L/16", 224): 0.40,
}

# 固定基座开销 (GB)：CUDA context + 模型权重 + 优化器状态 + 固定缓冲区
# 与 batch_size 无关，仅与模型变体有关。B 系列约 1.5GB，L 系列约 2.0GB。
BASE_OVERHEAD_GB_TABLE = {
    "JiT-B/16": 1.5,
    "JiT-L/16": 2.0,
}


# ---------------------------------------------------------------------------
# 硬件检测
# ---------------------------------------------------------------------------
def detect_hardware():
    """返回当前机器的硬件信息字典。"""
    info = {
        "cuda": torch.cuda.is_available(),
        "device_name": "cpu",
        "gpu_name": None,
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "cpu_cores": os.cpu_count() or 1,
    }
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = "cuda"
        info["gpu_name"] = props.name
        info["total_vram_gb"] = props.total_memory / 1024 ** 3
        try:
            info["free_vram_gb"] = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
        except Exception:
            info["free_vram_gb"] = 0.0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device_name"] = "mps"
    return info


def recommend_num_workers(cpu_cores, cap=12, min_workers=2):
    """根据 CPU 核数推荐 DataLoader 的 num_workers。"""
    if cpu_cores is None or cpu_cores <= 1:
        return 0
    return max(min_workers, min(cpu_cores - 1, cap))


def _round_batch(batch, multiple=4, min_val=1, max_batch=128):
    """把 batch 取整到 multiple 的倍数并夹在 [min_val, max_batch]。"""
    batch = int(batch)
    if batch <= multiple:
        return min_val
    batch = (batch // multiple) * multiple
    return max(min_val, min(batch, max_batch))


# ---------------------------------------------------------------------------
# table 模式：静态标定表估算
# ---------------------------------------------------------------------------
def estimate_batch_from_table(model_name, img_size, total_vram_gb, reserve=0.85,
                              base_overhead_gb=None, max_batch=128):
    """基于预标定的单样本显存常数，估算安全 batch_size。"""
    per_sample = PER_SAMPLE_GB_TABLE.get((model_name, img_size))
    if per_sample is None:
        raise ValueError(
            f"模型变体/分辨率 ({model_name}, {img_size}) 不在标定表中。"
            "请改用 method='probe' 进行实测，或手动设置 BATCH_SIZE。"
        )
    if base_overhead_gb is None:
        base_overhead_gb = BASE_OVERHEAD_GB_TABLE.get(model_name, 1.5)
    usable_gb = max(total_vram_gb * reserve - base_overhead_gb, 0.0)
    batch = int(usable_gb / per_sample)
    return _round_batch(batch, min_val=1, max_batch=max_batch), per_sample


# ---------------------------------------------------------------------------
# probe 模式：在真实模型上实测单样本显存增量
# ---------------------------------------------------------------------------
def probe_batch_size(model, device, img_size=256, ref_size=128,
                     num_fonts=1000, num_chars=20000, reserve=0.85,
                     max_batch=128, probe_batches=(1, 4), verbose=True):
    """在（已注入 LoRA 的）真实模型上实测训练式前向+反向的显存占用。

    返回 (safe_batch, per_sample_gb)；若无法在 CUDA 上探测则返回 (None, None)。
    该方法能自动捕获 LoRA r/alpha、模型变体、img_size、encoder 等任何参数
    对显存的影响，比静态表格更准确。

    注意：探测会短暂执行 1 次/4 次 batch 的训练前向+反向，耗时约数十秒
    （含 torch.compile 首次编译），之后会释放所有探测张量。
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        if verbose:
            print("[auto_tune] 非 CUDA 环境，跳过 probe 实测。")
        return None, None

    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    was_training = model.training
    model.train()

    def make_batch(bsz):
        x = torch.randn(bsz, 3, img_size, img_size, dtype=torch.float32, device=device) * 0.5
        labels = (
            torch.zeros(bsz, dtype=torch.long, device=device),
            torch.zeros(bsz, dtype=torch.long, device=device),
            torch.randn(bsz, 3, ref_size, ref_size, dtype=torch.float32, device=device) * 0.5,
            torch.randn(bsz, 3, img_size, img_size, dtype=torch.float32, device=device) * 0.5,
        )
        return x, labels

    def run_probe(bsz):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        x, labels = make_batch(bsz)
        with torch.amp.autocast("cuda", dtype=get_amp_dtype()):
            loss = model(x, labels)
        loss.backward()
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        model.zero_grad(set_to_none=True)
        del x, labels, loss
        torch.cuda.empty_cache()
        return peak_gb

    try:
        # 预热一次（触发 torch.compile 编译 + 分配器热身），丢弃其峰值
        x, labels = make_batch(probe_batches[0])
        with torch.amp.autocast("cuda", dtype=get_amp_dtype()):
            loss = model(x, labels)
        loss.backward()
        model.zero_grad(set_to_none=True)
        del x, labels, loss
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        model.train(was_training)
        return 1, None

    peak_b1 = None
    peak_b2 = None
    try:
        peak_b1 = run_probe(probe_batches[0])
    except torch.cuda.OutOfMemoryError:
        peak_b1 = None
    if peak_b1 is not None:
        try:
            peak_b2 = run_probe(probe_batches[1])
        except torch.cuda.OutOfMemoryError:
            peak_b2 = None

    if peak_b1 is not None and peak_b2 is not None:
        per_sample = max((peak_b2 - peak_b1) / (probe_batches[1] - probe_batches[0]), 0.01)
        overhead = peak_b1 - per_sample * probe_batches[0]
    elif peak_b1 is not None:
        # 只有 batch=1 成功：粗估单样本约为其一半
        per_sample = max(peak_b1 * 0.5, 0.01)
        overhead = peak_b1 - per_sample
    else:
        model.train(was_training)
        return 1, None

    safe_batch = int((total_vram_gb * reserve - overhead) / per_sample)
    safe_batch = _round_batch(safe_batch, min_val=1, max_batch=max_batch)
    model.train(was_training)
    return safe_batch, per_sample


def recommend_gen_bsz(batch_size, max_gen_bsz=32):
    """推理批量：推理无反向，但 CFG 会双倍前向，取保守值。"""
    if batch_size is None:
        return 16
    return max(1, min(batch_size, max_gen_bsz))


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def auto_tune(model=None, device=None, model_name="JiT-B/16", img_size=256,
              ref_size=128, num_fonts=1000, num_chars=20000, method="probe",
              reserve=0.85, base_overhead_gb=None, batch_size_fallback=16,
              max_batch=128, max_gen_bsz=32, num_workers_cap=12, verbose=True):
    """自动推算 batch_size / num_workers / gen_bsz。

    Args:
        model: 已构建并注入 LoRA 的模型（probe 模式需要；table 模式可传 None）。
        device: torch.device；为 None 时自动选择 cuda。
        method: "probe"（实测，推荐）或 "table"（标定表估算）。
        其余参数与各函数同名。
    Returns:
        dict: {batch_size, num_workers, gen_bsz, per_sample_gb, total_vram_gb, method}
    """
    hw = detect_hardware()
    if device is None:
        device = torch.device("cuda" if hw["cuda"] else "cpu")

    if not hw["cuda"]:
        if verbose:
            print("[auto_tune] 未检测到 CUDA，使用回退 batch_size={}。".format(batch_size_fallback))
        return {
            "batch_size": batch_size_fallback,
            "num_workers": recommend_num_workers(hw["cpu_cores"], cap=num_workers_cap),
            "gen_bsz": recommend_gen_bsz(batch_size_fallback, max_gen_bsz),
            "per_sample_gb": None,
            "total_vram_gb": 0.0,
            "method": "fallback",
        }

    per_sample = None
    if method == "probe" and model is not None:
        safe_batch, per_sample = probe_batch_size(
            model, device, img_size=img_size, ref_size=ref_size,
            num_fonts=num_fonts, num_chars=num_chars, reserve=reserve,
            max_batch=max_batch, verbose=verbose,
        )
        used_method = "probe"
        if safe_batch is None:
            # probe 失败（例如连 batch=1 都 OOM）-> 回退 table
            safe_batch, per_sample = estimate_batch_from_table(
                model_name, img_size, hw["total_vram_gb"], reserve=reserve,
                base_overhead_gb=base_overhead_gb, max_batch=max_batch,
            )
            used_method = "table(fallback)"
    else:
        safe_batch, per_sample = estimate_batch_from_table(
            model_name, img_size, hw["total_vram_gb"], reserve=reserve,
            base_overhead_gb=base_overhead_gb, max_batch=max_batch,
        )
        used_method = "table"

    num_workers = recommend_num_workers(hw["cpu_cores"], cap=num_workers_cap)
    gen_bsz = recommend_gen_bsz(safe_batch, max_gen_bsz)

    if verbose:
        print("=" * 56)
        print("  硬件自动调优结果 (method={})".format(used_method))
        print("=" * 56)
        print("  GPU:            {}".format(hw["gpu_name"] or "N/A"))
        print("  总显存:         {:.1f} GB".format(hw["total_vram_gb"]))
        print("  CPU 核数:       {}".format(hw["cpu_cores"]))
        print("  单样本显存:     {:.3f} GB (估算)".format(per_sample) if per_sample else "  单样本显存:     N/A")
        print("  batch_size:     {}".format(safe_batch))
        print("  gen_bsz:        {}".format(gen_bsz))
        print("  num_workers:    {}".format(num_workers))
        print("=" * 56)

    return {
        "batch_size": safe_batch,
        "num_workers": num_workers,
        "gen_bsz": gen_bsz,
        "per_sample_gb": per_sample,
        "total_vram_gb": hw["total_vram_gb"],
        "method": used_method,
    }
