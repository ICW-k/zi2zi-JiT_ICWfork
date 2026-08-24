import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image

from .charsets import SUPPORTED_CHARSETS, get_charset_codepoints
from .font_utils import (
    GlyphRenderer,
    ensure_output_directory,
    extract_font_name,
    get_cjk_codepoints,
    load_font,
)


DEFAULT_IMAGE_RESOLUTION = 256
DEFAULT_JPEG_QUALITY = 95


def _format_codepoint(codepoint: int) -> str:
    return f"U+{codepoint:04X}"


def _parse_codepoint(codepoint: str) -> int:
    text = codepoint.strip().upper()
    if text.startswith("U+"):
        text = text[2:]
    return int(text, 16)


def _build_index_map(charset: Optional[str]) -> Dict[int, int]:
    if not charset:
        return {}
    codepoints = sorted(get_charset_codepoints(charset))
    return {cp: idx for idx, cp in enumerate(codepoints)}


def _save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def create_reference_grid(
    renderer: GlyphRenderer,
    ref_codepoints: List[int],
    grid_size: int = 256,
    cell_size: int = 128,
) -> Optional[Image.Image]:
    if len(ref_codepoints) != 4:
        raise ValueError("ref_codepoints must contain exactly 4 codepoints")

    grid = Image.new("RGB", (grid_size, grid_size), (255, 255, 255))
    positions = [(0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size)]

    for idx, codepoint in enumerate(ref_codepoints):
        glyph = renderer.render(codepoint)
        if glyph is None:
            return None
        glyph = glyph.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        grid.paste(glyph, positions[idx])
    return grid


def create_combined_image(
    source_img: Image.Image,
    target_img: Image.Image,
    ref_grid_1: Image.Image,
    ref_grid_2: Image.Image,
) -> Image.Image:
    combined = Image.new("RGB", (1024, 256), (255, 255, 255))
    combined.paste(source_img, (0, 0))
    combined.paste(target_img, (256, 0))
    combined.paste(ref_grid_1, (512, 0))
    combined.paste(ref_grid_2, (768, 0))
    return combined


def _font_metadata(
    target_font_path: Path,
    target_font,
    target_codepoints: set[int],
    source_font_path: Path,
    charset: Optional[str],
    font_index: Optional[int],
) -> dict:
    meta = {
        "font_file": target_font_path.name,
        "font_name": extract_font_name(target_font, target_font_path),
        "total_cjk_glyphs": len(target_codepoints),
        "source_font": source_font_path.name,
        "charset_filter": charset,
        "extraction_date": datetime.now().isoformat(),
    }
    if font_index is not None:
        meta["font_index"] = font_index
    return meta


def _sample_codepoints(codepoints: List[int], sample_count: int, seed: Optional[int]) -> List[int]:
    if sample_count >= len(codepoints):
        return list(codepoints)
    rng = random.Random(seed)
    selected = rng.sample(codepoints, sample_count)
    selected.sort()
    return selected


def _pick_refs(reference_pool: List[int], current_codepoint: int, seed: Optional[int]) -> Optional[List[int]]:
    pool = [cp for cp in reference_pool if cp != current_codepoint]
    if len(pool) < 8:
        return None
    rng = random.Random((seed or 0) + current_codepoint)
    return rng.sample(pool, 8)


def _filename_for_codepoint(codepoint: int, local_index: int) -> str:
    return f"{local_index:05d}_{_format_codepoint(codepoint)}.jpg"


def extract_train_src_target_refs(
    source_font_path: Path,
    target_font_path: Path,
    output_dir: Path,
    sample_count: int = 500,
    charset: str = "gb2312",
    resolution: int = DEFAULT_IMAGE_RESOLUTION,
    jpg_quality: int = DEFAULT_JPEG_QUALITY,
    seed: Optional[int] = None,
    font_index: Optional[int] = None,
) -> dict:
    source_font, source_validated_path = load_font(str(source_font_path))
    target_font, target_validated_path = load_font(str(target_font_path))

    source_codepoints = get_cjk_codepoints(source_font)
    target_codepoints = get_cjk_codepoints(target_font)
    common_codepoints = source_codepoints & target_codepoints

    if charset:
        if charset.lower() not in SUPPORTED_CHARSETS:
            raise ValueError(f"Unsupported charset '{charset}'. Supported: {', '.join(sorted(SUPPORTED_CHARSETS))}")
        charset_codepoints = get_charset_codepoints(charset)
        filtered_codepoints = common_codepoints & charset_codepoints
        index_map = _build_index_map(charset)
    else:
        filtered_codepoints = common_codepoints
        index_map = {}

    if len(filtered_codepoints) < 9:
        return {
            "success": False,
            "error": f"Not enough matching glyphs found (need at least 9, got {len(filtered_codepoints)})",
            "font_file": target_font_path.name,
        }

    selected_codepoints = _sample_codepoints(sorted(filtered_codepoints), sample_count, seed)
    ensure_output_directory(str(output_dir))

    source_renderer = GlyphRenderer(str(source_validated_path), resolution)
    target_renderer = GlyphRenderer(str(target_validated_path), resolution)

    successful = 0
    failed = 0
    extracted_chars = []

    for local_index, codepoint in enumerate(selected_codepoints):
        source_img = source_renderer.render(codepoint)
        target_img = target_renderer.render(codepoint)
        if source_img is None or target_img is None:
            failed += 1
            continue

        refs = _pick_refs(selected_codepoints, codepoint, seed)
        if refs is None:
            failed += 1
            continue

        ref_grid_1 = create_reference_grid(target_renderer, refs[:4])
        ref_grid_2 = create_reference_grid(target_renderer, refs[4:])
        if ref_grid_1 is None or ref_grid_2 is None:
            failed += 1
            continue

        combined = create_combined_image(source_img, target_img, ref_grid_1, ref_grid_2)
        filename = _filename_for_codepoint(codepoint, local_index)
        combined.save(output_dir / filename, "JPEG", quality=jpg_quality)

        extracted_chars.append(
            {
                "codepoint": _format_codepoint(codepoint),
                "character": chr(codepoint),
                "filename": filename,
                "charset_index": index_map.get(codepoint),
                "reference_codepoints_1": [_format_codepoint(cp) for cp in refs[:4]],
                "reference_codepoints_2": [_format_codepoint(cp) for cp in refs[4:]],
            }
        )
        successful += 1
    logger.info("  train: %d extracted, %d failed", successful, failed)

    metadata = _font_metadata(
        target_font_path=target_font_path,
        target_font=target_font,
        target_codepoints=target_codepoints,
        source_font_path=source_font_path,
        charset=charset,
        font_index=font_index,
    )
    metadata.update(
        {
            "dataset_type": "train",
            "sample_requested": sample_count,
            "extracted_count": successful,
            "failed_count": failed,
            "resolution": resolution,
            "image_dimensions": "1024x256",
            "layout": "source(256) + target(256) + refs_grid1(256) + refs_grid2(256)",
            "characters": extracted_chars,
        }
    )
    _save_json(output_dir / "metadata.json", metadata)

    return {
        "success": True,
        "font_file": target_font_path.name,
        "font_index": font_index,
        "extracted": successful,
        "failed": failed,
        "output_dir": str(output_dir),
    }


def load_training_codepoints(train_metadata_path: Path) -> List[int]:
    with open(train_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    codepoints = []
    for char in metadata.get("characters", []):
        cp = _parse_codepoint(char["codepoint"])
        codepoints.append(cp)
    return codepoints


def extract_test_src_target_refs(
    source_font_path: Path,
    target_font_path: Path,
    output_dir: Path,
    train_codepoints: List[int],
    test_sample_count: int = 8,
    charset: str = "gb2312",
    resolution: int = DEFAULT_IMAGE_RESOLUTION,
    jpg_quality: int = DEFAULT_JPEG_QUALITY,
    seed: Optional[int] = None,
    font_index: Optional[int] = None,
) -> dict:
    source_font, source_validated_path = load_font(str(source_font_path))
    target_font, target_validated_path = load_font(str(target_font_path))

    source_codepoints = get_cjk_codepoints(source_font)
    target_codepoints = get_cjk_codepoints(target_font)
    common_codepoints = source_codepoints & target_codepoints

    if charset:
        if charset.lower() not in SUPPORTED_CHARSETS:
            raise ValueError(f"Unsupported charset '{charset}'. Supported: {', '.join(sorted(SUPPORTED_CHARSETS))}")
        charset_codepoints = get_charset_codepoints(charset)
        filtered_codepoints = common_codepoints & charset_codepoints
        index_map = _build_index_map(charset)
    else:
        filtered_codepoints = common_codepoints
        index_map = {}

    train_set = set(train_codepoints)
    unseen_codepoints = sorted(filtered_codepoints - train_set)
    degraded_test = False

    if len(unseen_codepoints) < test_sample_count:
        # 训练已覆盖(几乎)全部交集字（TRAIN_CHARS_PER_FONT >= 交集大小 时必然发生），
        # 无法再挑"训练未见字"。降级为从全部交集字中抽展示用字（含训练字）：
        # 仅用于可视化还原，不作为"未见字泛化"指标。
        if len(filtered_codepoints) < test_sample_count:
            return {
                "success": False,
                "error": f"Not enough unseen glyphs (need {test_sample_count}, got {len(unseen_codepoints)})",
                "font_file": target_font_path.name,
            }
        selected_codepoints = _sample_codepoints(sorted(filtered_codepoints), test_sample_count, seed)
        degraded_test = True
    else:
        selected_codepoints = _sample_codepoints(unseen_codepoints, test_sample_count, seed)
    if len(train_codepoints) < 8:
        return {
            "success": False,
            "error": f"Not enough training references (need 8, got {len(train_codepoints)})",
            "font_file": target_font_path.name,
        }

    ensure_output_directory(str(output_dir))
    source_renderer = GlyphRenderer(str(source_validated_path), resolution)
    target_renderer = GlyphRenderer(str(target_validated_path), resolution)

    successful = 0
    failed = 0
    extracted_chars = []

    for local_index, codepoint in enumerate(selected_codepoints):
        source_img = source_renderer.render(codepoint)
        target_img = target_renderer.render(codepoint)
        if source_img is None or target_img is None:
            failed += 1
            continue

        rng = random.Random((seed or 0) + codepoint)
        refs = rng.sample(train_codepoints, 8)

        ref_grid_1 = create_reference_grid(target_renderer, refs[:4])
        ref_grid_2 = create_reference_grid(target_renderer, refs[4:])
        if ref_grid_1 is None or ref_grid_2 is None:
            failed += 1
            continue

        combined = create_combined_image(source_img, target_img, ref_grid_1, ref_grid_2)
        filename = _filename_for_codepoint(codepoint, local_index)
        combined.save(output_dir / filename, "JPEG", quality=jpg_quality)

        extracted_chars.append(
            {
                "codepoint": _format_codepoint(codepoint),
                "character": chr(codepoint),
                "filename": filename,
                "charset_index": index_map.get(codepoint),
                "reference_codepoints_1": [_format_codepoint(cp) for cp in refs[:4]],
                "reference_codepoints_2": [_format_codepoint(cp) for cp in refs[4:]],
            }
        )
        successful += 1
    logger.info("  test: %d extracted, %d failed", successful, failed)

    metadata = _font_metadata(
        target_font_path=target_font_path,
        target_font=target_font,
        target_codepoints=target_codepoints,
        source_font_path=source_font_path,
        charset=charset,
        font_index=font_index,
    )
    metadata.update(
        {
            "dataset_type": "test",
            "sample_requested": test_sample_count,
            "training_reference_count": len(train_codepoints),
            "note": ("Test set: target characters are UNSEEN, references are from training set"
                     if not degraded_test else
                     "DEGRADED: train covered all overlap chars; test chars are seen (visualization only)"),
            "extracted_count": successful,
            "failed_count": failed,
            "resolution": resolution,
            "image_dimensions": "1024x256",
            "layout": "source(256) + target(256) + refs_grid1(256) + refs_grid2(256)",
            "characters": extracted_chars,
        }
    )
    _save_json(output_dir / "metadata.json", metadata)

    return {
        "success": True,
        "font_file": target_font_path.name,
        "font_index": font_index,
        "extracted": successful,
        "failed": failed,
        "output_dir": str(output_dir),
        "degraded_test": degraded_test,
    }


def _list_font_files(font_dir: Path) -> List[Path]:
    suffixes = {".ttf", ".otf", ".TTF", ".OTF"}
    return sorted([p for p in font_dir.iterdir() if p.is_file() and p.suffix in suffixes], key=lambda p: p.name.lower())


def _train_one_font(args_tuple):
    (source_font, target_font, font_output_dir, chars_per_font,
     charset, resolution, seed, font_index, label) = args_tuple
    print(label, flush=True)
    result = extract_train_src_target_refs(
        source_font_path=source_font,
        target_font_path=target_font,
        output_dir=font_output_dir,
        sample_count=chars_per_font,
        charset=charset,
        resolution=resolution,
        seed=seed + font_index,
        font_index=font_index,
    )
    print(f"  done: extracted={result.get('extracted', 0)} failed={result.get('failed', 0)}", flush=True)
    return font_index, result


def generate_train_dataset(
    source_font: Path,
    font_dir: Path,
    output_dir: Path,
    num_fonts: Optional[int] = None,
    chars_per_font: int = 500,
    charset: str = "gb2312",
    resolution: int = DEFAULT_IMAGE_RESOLUTION,
    seed: int = 42,
    start_index: int = 1,
    num_workers: int = 1,
) -> dict:
    ensure_output_directory(str(output_dir))
    font_files = _list_font_files(font_dir)
    if num_fonts is not None:
        font_files = font_files[:num_fonts]

    total_fonts = len(font_files)
    tasks = []
    for offset, target_font in enumerate(font_files):
        font_index = start_index + offset
        folder_name = f"{font_index:03d}_{target_font.stem}"
        font_output_dir = output_dir / folder_name
        label = f"[{offset + 1}/{total_fonts}] {target_font.name}"
        tasks.append((source_font, target_font, font_output_dir, chars_per_font,
                       charset, resolution, seed, font_index, label))

    result_map = {}
    if num_workers <= 1:
        for t in tasks:
            font_index, result = _train_one_font(t)
            result_map[font_index] = result
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_train_one_font, t): t for t in tasks}
            for future in as_completed(futures):
                font_index, result = future.result()
                result_map[font_index] = result

    results = [result_map[start_index + i] for i in range(len(font_files))]
    success = sum(1 for r in results if r.get("success"))
    failed = len(results) - success
    return {"total": len(results), "success": success, "failed": failed, "results": results}


def _resolve_target_font(train_folder: Path, train_metadata: dict, font_dir: Path) -> Optional[Path]:
    font_file = train_metadata.get("font_file")
    if isinstance(font_file, str):
        candidate = font_dir / font_file
        if candidate.exists():
            return candidate

    suffix = train_folder.name.split("_", 1)
    if len(suffix) == 2:
        stem = suffix[1]
    else:
        stem = train_folder.name

    for ext in (".TTF", ".ttf", ".OTF", ".otf"):
        candidate = font_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _test_one_font(args_tuple):
    (offset, source_font, target_font, font_output_dir, train_codepoints,
     chars_per_font, charset, resolution, seed, font_index, label) = args_tuple
    print(label, flush=True)
    result = extract_test_src_target_refs(
        source_font_path=source_font,
        target_font_path=target_font,
        output_dir=font_output_dir,
        train_codepoints=train_codepoints,
        test_sample_count=chars_per_font,
        charset=charset,
        resolution=resolution,
        seed=seed,
        font_index=font_index,
    )
    print(f"  done: extracted={result.get('extracted', 0)} failed={result.get('failed', 0)}", flush=True)
    return offset, result


def generate_test_dataset(
    source_font: Path,
    font_dir: Path,
    train_dir: Path,
    output_dir: Path,
    chars_per_font: int = 8,
    charset: str = "gb2312",
    resolution: int = DEFAULT_IMAGE_RESOLUTION,
    seed: int = 99999,
    num_workers: int = 1,
) -> dict:
    ensure_output_directory(str(output_dir))
    train_folders = sorted([p for p in train_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    total_fonts = len(train_folders)
    tasks = []
    result_map = {}
    for offset, train_folder in enumerate(train_folders):
        label = f"[{offset + 1}/{total_fonts}] {train_folder.name}"
        metadata_path = train_folder / "metadata.json"
        if not metadata_path.exists():
            result_map[offset] = {
                "success": False,
                "font_file": train_folder.name,
                "error": "metadata.json not found in training folder",
            }
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            train_metadata = json.load(f)

        target_font = _resolve_target_font(train_folder, train_metadata, font_dir)
        if target_font is None:
            result_map[offset] = {
                "success": False,
                "font_file": train_folder.name,
                "error": "target font file not found",
            }
            continue

        train_codepoints = load_training_codepoints(metadata_path)
        font_index = train_metadata.get("font_index")
        font_output_dir = output_dir / train_folder.name

        tasks.append((offset, source_font, target_font, font_output_dir, train_codepoints,
                       chars_per_font, charset, resolution, seed + offset,
                       font_index if isinstance(font_index, int) else None, label))

    if num_workers <= 1:
        for t in tasks:
            off, result = _test_one_font(t)
            result_map[off] = result
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_test_one_font, t): t for t in tasks}
            for future in as_completed(futures):
                off, result = future.result()
                result_map[off] = result

    results = [result_map[i] for i in range(total_fonts)]
    success = sum(1 for r in results if r.get("success"))
    failed = len(results) - success
    return {"total": len(results), "success": success, "failed": failed, "results": results}


def _extract_ref(img, ref_global_idx, ref_size):
    grid_idx = ref_global_idx // 4
    ref_idx = ref_global_idx % 4
    grid_x = 512 if grid_idx == 0 else 768
    row = ref_idx // 2
    col = ref_idx % 2
    x1 = grid_x + col * 128
    y1 = row * 128
    ref = img.crop((x1, y1, x1 + 128, y1 + 128))
    if ref.size[0] != ref_size or ref.size[1] != ref_size:
        ref = ref.resize((ref_size, ref_size), Image.LANCZOS)
    return ref


def _collect_test_samples(root):
    font_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')
    ])
    samples = []
    for font_name in font_dirs:
        font_path = os.path.join(root, font_name)
        idx_str = font_name.split('_')[0]
        if idx_str.startswith("'"):
            idx_str = idx_str[1:]
        font_idx = int(idx_str) - 1
        font_samples = []
        for filename in os.listdir(font_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            try:
                parts = filename.split('_')
                char_idx = int(parts[0])
            except (ValueError, IndexError):
                continue
            unicode_cp = 0
            for part in parts[1:]:
                part_clean = os.path.splitext(part)[0]
                if part_clean.startswith('U+'):
                    unicode_cp = int(part_clean[2:], 16)
                    break
            font_samples.append((os.path.join(font_path, filename), font_idx, char_idx, unicode_cp))
        font_samples.sort(key=lambda s: (s[1], s[2]))
        samples.extend(font_samples)
    samples.sort(key=lambda s: (s[1], s[2]))
    return samples


def create_test_npz(data_path, output_path, ref_index=0, ref_size=128):
    """Convert a test dataset folder (1024x256 composites) into an .npz file."""
    data_path, output_path = str(data_path), str(output_path)
    samples = _collect_test_samples(data_path)
    n = len(samples)
    if n == 0:
        return {"samples": 0, "file_size_mb": 0}
    font_labels = np.empty(n, dtype=np.int64)
    char_labels = np.empty(n, dtype=np.int64)
    unicode_labels = np.empty(n, dtype=np.int64)
    content_images = np.empty((n, 3, 256, 256), dtype=np.uint8)
    target_images = np.empty((n, 3, 256, 256), dtype=np.uint8)
    style_images = np.empty((n, 3, ref_size, ref_size), dtype=np.uint8)
    for i, (img_path, font_idx, char_idx, unicode_cp) in enumerate(samples):
        img = Image.open(img_path).convert('RGB')
        source = img.crop((0, 0, 256, 256))
        target = img.crop((256, 0, 512, 256))
        ref = _extract_ref(img, ref_index, ref_size)
        font_labels[i] = font_idx
        char_labels[i] = char_idx
        unicode_labels[i] = unicode_cp
        content_images[i] = np.array(source).transpose(2, 0, 1)
        target_images[i] = np.array(target).transpose(2, 0, 1)
        style_images[i] = np.array(ref).transpose(2, 0, 1)
    logger.info("  npz: %d samples processed", n)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(
        output_path,
        font_labels=font_labels,
        char_labels=char_labels,
        unicode_labels=unicode_labels,
        content_images=content_images,
        target_images=target_images,
        style_images=style_images,
        num_original_samples=np.int64(n),
    )
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    return {"samples": n, "file_size_mb": file_size}
