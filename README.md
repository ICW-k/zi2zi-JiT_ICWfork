<p align="center">
  <img src="assets/logo.svg" alt="zi2zi-JiT logo" width="200">
</p>

<h2 align="center">zi2zi-JiT: Font Synthesis with Pixel Space Diffusion Transformers</h2>

<p align="center">
  <a href="README_zh.md">中文版</a>
</p>

<p align="center">
  <img src="assets/showcase.png" width="80%">
</p>

## Overview

<p align="center">
  <img src="assets/overview.png" width="100%">
</p>

zi2zi-JiT is a conditional variant of [JiT](https://arxiv.org/abs/2511.13720) (Just image Transformer) designed for Chinese font style transfer. Given a source character and a style reference, it synthesizes the character in the target font style.

The architecture, illustrated above, extends the base JiT model with three components:

- **Content Encoder** — a CNN that captures the structural layout of the input character, adapted from [FontDiffuser](https://arxiv.org/abs/2312.12142).
- **Style Encoder** — a CNN that extracts stylistic features from a reference glyph in the target font.
- **Multi-Source In-Context Mixing** — instead of conditioning on a single category token as in the original JiT, font, style, and content embeddings are concatenated into a unified conditioning sequence.

### Training

Two model variants are available — JiT-B/16 and JiT-L/16 — both trained for 2,000 epochs on a corpus of over 400+ fonts (70% simplified Chinese, 20% traditional Chinese, 10% Japanese), totalling 300k+ character images. For each font, the max number of characters used for training is capped at 800

### Evaluation

Generated glyphs are evaluated against ground-truth references following the protocol in [FontDiffuser](https://arxiv.org/abs/2312.12142). All metrics are computed over 2,400 pairs.

| Model | FID ↓ | SSIM ↑ | LPIPS ↓ | L1 ↓ |
|-------|-------|--------|---------|------|
| JiT-B/16 | 53.81 | 0.6753 | 0.2024 | 0.1071 |
| JiT-L/16 | 56.01 | 0.6794 | 0.1967 | 0.1043 |

## How To Use

### Environment Setup

```bash
conda env create -f environment.yaml
conda activate zi2zi-jit
pip install -e .
```

### Download

Pretrained checkpoints are available on Google Drive:

**[Download Models](https://drive.google.com/drive/folders/1QJi2ihxDBK2NF-jCE07g59YwuUTAd-iY)**

Save desired checkpoint and place it under `models/`:

```bash
mkdir -p models
# zi2zi-JiT-B-16.pth  (Base variant)
# zi2zi-JiT-L-16.pth  (Large variant)
```

### Dataset Generation

#### From font files

Generate a paired dataset from a source font and a directory of target fonts:

```bash
python scripts/generate_font_dataset.py \
    --source-font data/思源宋体light.otf \
    --font-dir   data/sample_single_font \
    --output-dir data/sample_dataset
```

This produces the following structure:

```
data/sample_dataset/
├── train/
│   ├── 001_FontA/
│   │   ├── 00000_U+XXXX.jpg
│   │   ├── 00001_U+XXXX.jpg
│   │   ├── ...
│   │   └── metadata.json
│   ├── 002_FontB/
│   │   └── ...
│   └── ...
├── test/
│   ├── 001_FontA/
│   │   └── ...
│   └── ...
└── test.npz
```

Each `.jpg` is a 1024x256 composite: `source (256) | target (256) | ref_grid_1 (256) | ref_grid_2 (256)`.

#### From rendered glyph images

Alternatively, build a dataset from a directory of rendered character images.
Each file should be a 256x256 PNG named by its character:

```
data/sample_glyphs/
├── 万.png
├── 上.png
├── 中.png
├── 人.png
├── 大.png
└── ...
```

```bash
python scripts/generate_glyph_dataset.py \
    --source-font data/思源宋体light.otf \
    --glyph-dir   data/sample_glyphs \
    --output-dir  data/sample_glyph_dataset \
    --train-count 200
```

### LoRA Fine-Tuning

Fine-tune a pretrained model on a single GPU with LoRA. Fine-tuning a single font typically takes less than one hour on a single H100. The example below uses JiT-B/16 with batch size 16, which requires roughly 4 GB of VRAM:

```bash
python lora_single_gpu_finetune_jit.py \
    --data_path       data/sample_dataset/train/ \
    --test_npz_path   data/sample_dataset/test.npz \
    --output_dir      run/lora_ft_sample_single/ \
    --base_checkpoint models/zi2zi-JiT-B-16.pth \
    --model           JiT-B/16 \
    --num_fonts       1000 \
    --num_chars       20000 \
    --max_chars_per_font 200 \
    --img_size        256 \
    --lora_r          32 \
    --lora_alpha      32 \
    --lora_targets    "qkv,proj,w12,w3" \
    --epochs          200 \
    --batch_size      16 \
    --blr             8e-4 \
    --warmup_epochs   1 \
    --save_last_freq  10 \
    --proj_dropout    0.1 \
    --P_mean          -0.8 \
    --P_std           0.8 \
    --noise_scale     1.0 \
    --cfg             2.6 \
    --online_eval \
    --eval_step_folders \
    --eval_freq       10 \
    --gen_bsz         16 \
    --num_images      400 \
    --seed            42
```

**Key parameters:**

| Parameter | Note |
|---|---|
| `--num_fonts`, `--num_chars` | Tied to the pretrained model's embedding size. Do not change unless pretraining from scratch. |
| `--max_chars_per_font` | Caps the number of characters used from each font. |
| `--lora_r`, `--lora_alpha` | LoRA capacity. Higher values give more capacity at the cost of memory. |
| `--batch_size` | 16 uses ~4 GB VRAM. |
| `--cfg` | Conditioning strength. Use **2.6** for JiT-B/16, **2.4** for JiT-L/16. |

### Generation

Generate characters from a fine-tuned checkpoint:

```bash
python generate_chars.py \
    --checkpoint run/lora_ft_sample_single/checkpoint-last.pth \
    --test_npz   data/sample_dataset/test.npz \
    --output_dir run/generated_chars/
```

**Notes for `generate_chars.py`:**

- Supported samplers are `euler`, `heun`, and `ab2`.
- If `--num_sampling_steps` is not set, the script uses method-specific defaults:
  `euler -> 20`, `heun -> 50`, `ab2 -> 20`.
- If neither `--sampling_method` nor `--num_sampling_steps` is overridden, the script keeps the checkpoint's saved inference settings.
- Current recommended fast setting: `--sampling_method ab2 --cfg 2.6` and let the default `20` steps apply.
- `heun-50` is kept as a conservative legacy/reference baseline. In the current 50-sample MPS benchmark, `ab2-20` and `euler-20` were both faster and scored better than `heun-50` on SSIM, LPIPS, and L1.
- `--pairwise` saves side-by-side comparisons: `src_gen` (source|generated) or `target_gen` (target|generated, requires `target_images` in the npz). Comparisons go to the `compare/` subfolder, ready for `compute_pairwise_metrics.py`.

Example fast generation command:

```bash
python generate_chars.py \
    --checkpoint run/lora_ft_sample_single/checkpoint-last.pth \
    --test_npz   data/sample_dataset/test.npz \
    --output_dir run/generated_chars_ab2/ \
    --sampling_method ab2
```

### Missing Glyph Completion

When the target font lacks some characters in the charset (e.g. `gb2312`), a fine-tuned LoRA checkpoint can complete them:

1. Read the target font's cmap with fontTools and compute `charset - covered characters = missing characters`;
2. Missing characters are rendered by the **source font** as the content glyph (the model cannot invent unseen structures);
3. Style reference images come from the **target font itself** (matching the training ref grid);
4. Each missing glyph is generated as PNG and a manifest file is written.

```bash
python scripts/generate_missing_chars.py \
    --checkpoint run/lora_ft_sample_single/checkpoint-last.pth \
    --target-font fonts/<target_font>.ttf \
    --source-font fonts/<source_font>.ttf \
    --charset gb2312 \
    --output-dir run/lora_ft_sample_single/missing_chars
```

Output structure:

```
run/lora_ft_sample_single/missing_chars/
├── generated/          # Generated missing glyphs (0000_U+XXXX.png)
├── compare/            # Source|generated comparisons (only with --pairwise src_gen)
└── missing_chars.txt   # Manifest (U+XXXX\tcharacter)
```

**Key parameters:**

| Parameter | Note |
|---|---|
| `--target-font`, `--source-font` | Required. Target font (missing glyphs to complete) + source font (provides content glyphs, **must cover the missing characters**). |
| `--charset` | Charset used for completion, default `gb2312`. |
| `--pairwise` | Whether to save `source\|generated` side-by-side comparisons. `src_gen` (default, doubles disk I/O) / `none` (generated images only, saves half the I/O). `target_gen` is unavailable here — the target font has no such missing glyphs, and the script does not implement that branch. |
| `--ref-chars` | Comma-separated style reference characters (default: auto-picked from characters the target font can render). |
| `--ref-count` | Number of auto-picked style reference characters, default 8 (capped at 8). |
| `--cfg`, `--num-sampling-steps`, `--sampling-method` | Sampling parameters, defaults taken from the checkpoint (same rules as `generate_chars.py`). |
| `--resolution` | Render resolution, must match training (default 256). |
| `--num-images` | Max number of missing glyphs to generate, default all. |
| `--batch-size` | Inference batch size, default 64. |
| `--seed`, `--device` | Random seed and device (auto/cpu/cuda/mps). |

**Notes:**

- **Resume support**: glyphs already present in `generated/` are skipped; rerunning after an interruption only processes the remainder.
- **Source coverage**: missing characters the source font cannot render are skipped (see `missing_chars.txt`).
- **Embedding limit**: characters beyond the model's `num_chars` embedding space are skipped.
- **Training alignment**: to fully align character labels with training, train with `TRAIN_CHARS_PER_FONT` covering the whole CHARSET and `MAX_CHARS_PER_FONT` set to None.

### Metrics

Compute pairwise metrics (SSIM, LPIPS, L1, FID) on the generated comparison grids:

```bash
python scripts/compute_pairwise_metrics.py \
    --device cuda \
    run/lora_ft_sample_single/heun-steps50-cfg2.6-interval0.0-1.0-image400-res256/step_10/compare/
```

## Works

Fonts created with zi2zi-JiT:

- [Zi-QuanHengDuLiang (权衡度量体)](https://github.com/kaonashi-tyc/Zi-QuanHengDuLiang)
- [Zi-XuanZongTi (玄宗体)](https://github.com/kaonashi-tyc/Zi-XuanZongTi)
- [Eva-Ming-Simplified (Eva明朝简体)](https://github.com/kaonashi-tyc/Eva-Ming-Simplified)

## Gallery

Ground truth on the left, generated one on the right

| | |
|:---:|:---:|
| ![Cartoonish](assets/gallery/cartonish.png) | ![Cursive](assets/gallery/cursive.png) |
| ![Geometric](assets/gallery/geometric.png) | ![Thin](assets/gallery/thin.png) |
| ![Zhuan Shu](assets/gallery/zhuan_shu.png) | ![Brush](assets/gallery/brush.png) |

### License

Code is licensed under MIT. Generated font outputs are additionally subject to
the "Font Artifact License Addendum" in [LICENSE](LICENSE):

- commercial use is allowed
- attribution is required when distributing a font product that uses more
  than 200 characters created from repository artifacts

### References / Thanks

- [JiT: Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)
- [FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning](https://arxiv.org/abs/2312.12142) ([code](https://github.com/yeungchenwa/FontDiffuser))

This project builds on code and ideas from:

- [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) — content/style encoder design and evaluation protocol
- [JiT](https://github.com/LTH14/JiT) — base diffusion transformer architecture

### Citation

```bibtex
@article{zi2zi-jit,
  title   = {zi2zi-JiT: Font Synthesis with Pixel Space Diffusion Transformers},
  author  = {Yuchen Tian},
  year    = {2026},
  url     = {https://github.com/kaonashi-tyc/zi2zi-jit}
}
```
