#!/usr/bin/env python3
"""Generate FontSrcTarget train and/or test dataset from target TTF/OTF fonts.

By default generates both train/ and test/ under --output-dir.

Usage:
    # Both train and test
    python scripts/generate_font_dataset.py \
        --source-font /path/to/source.ttf \
        --font-dir /path/to/fonts/ \
        --output-dir /path/to/dataset

    # Train only
    python scripts/generate_font_dataset.py \
        --source-font /path/to/source.ttf \
        --font-dir /path/to/fonts/ \
        --output-dir /path/to/dataset \
        --train-only

    # Test only (requires existing train dir)
    python scripts/generate_font_dataset.py \
        --source-font /path/to/source.ttf \
        --font-dir /path/to/fonts/ \
        --output-dir /path/to/dataset \
        --test-only --train-dir /path/to/existing_train
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# 把仓库根目录加入 sys.path，脚本可在任意位置运行（例如被 subprocess 调用时）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_processing.pipeline import generate_train_dataset, generate_test_dataset, create_test_npz


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FontSrcTarget dataset (train + test by default)."
    )
    parser.add_argument("--source-font", required=True, help="Path to source/reference font.")
    parser.add_argument("--font-dir", required=True, help="Directory containing target fonts.")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory. Train goes to <output-dir>/train, test to <output-dir>/test.")

    # Mode flags
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true", help="Generate train dataset only.")
    mode.add_argument("--test-only", action="store_true", help="Generate test dataset only (requires --train-dir).")

    # Train args
    parser.add_argument("--num-fonts", type=int, default=None, help="Number of target fonts to process.")
    parser.add_argument("--start-index", type=int, default=1, help="Starting font index for folder names.")
    parser.add_argument("--train-chars-per-font", type=int, default=500,
                        help="Characters per font for training (default: 500).")
    parser.add_argument("--train-seed", type=int, default=42, help="Random seed for training (default: 42).")

    # Test args
    parser.add_argument("--train-dir", type=str, default=None,
                        help="Existing train directory (required for --test-only, ignored otherwise).")
    parser.add_argument("--test-chars-per-font", type=int, default=8,
                        help="Unseen characters per font for testing (default: 8).")
    parser.add_argument("--test-seed", type=int, default=99999, help="Random seed for testing (default: 99999).")

    # Shared
    parser.add_argument("--charset", type=str, default="gb2312",
                        choices=["gb2312", "gbk", "big5", "jisx0208", "ksx1001"],
                        help="Charset filter (default: gb2312).")
    parser.add_argument("--resolution", type=int, default=256, help="Glyph resolution (default: 256).")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers for font processing (default: 4).")

    args = parser.parse_args()

    if args.test_only and args.train_dir is None:
        parser.error("--test-only requires --train-dir")

    if args.test_only and not Path(args.train_dir).is_dir():
        raise FileNotFoundError(f"--train-dir does not exist: {args.train_dir}")

    return args


def print_summary(label: str, summary: dict) -> None:
    print(f"\n{label} generation completed: "
          f"total={summary['total']} success={summary['success']} failed={summary['failed']}")
    for result in summary["results"]:
        tag = "OK" if result.get("success") else "SKIP"
        font_file = result.get("font_file", "<unknown>")
        info = result.get("error", f"extracted={result.get('extracted', 0)} failed={result.get('failed', 0)}")
        print(f"  [{tag}] {font_file}: {info}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    output_dir = Path(args.output_dir)
    do_train = not args.test_only
    do_test = not args.train_only

    train_dir = Path(args.train_dir) if args.train_dir else output_dir / "train"

    if do_train:
        train_out = output_dir / "train"
        print(f"=== Generating train dataset -> {train_out} ===")
        train_summary = generate_train_dataset(
            source_font=Path(args.source_font),
            font_dir=Path(args.font_dir),
            output_dir=train_out,
            num_fonts=args.num_fonts,
            chars_per_font=args.train_chars_per_font,
            charset=args.charset,
            resolution=args.resolution,
            seed=args.train_seed,
            start_index=args.start_index,
            num_workers=args.num_workers,
        )
        print_summary("Train", train_summary)
        train_dir = train_out

    if do_test:
        test_out = output_dir / "test"
        print(f"\n=== Generating test dataset -> {test_out} ===")
        test_summary = generate_test_dataset(
            source_font=Path(args.source_font),
            font_dir=Path(args.font_dir),
            train_dir=train_dir,
            output_dir=test_out,
            chars_per_font=args.test_chars_per_font,
            charset=args.charset,
            resolution=args.resolution,
            seed=args.test_seed,
            num_workers=args.num_workers,
        )
        print_summary("Test", test_summary)

        # Convert test set to NPZ
        npz_path = output_dir / "test.npz"
        print(f"\n=== Creating test NPZ -> {npz_path} ===")
        result = create_test_npz(test_out, npz_path)
        print(f"  {result['samples']} samples ({result['file_size_mb']:.1f} MB)")


if __name__ == "__main__":
    main()
