from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drive_kd.datasets.manifest_utils import build_absolute_manifest_from_portable
from drive_kd.datasets.path_resolver import DatasetPathResolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local absolute manifests from portable Drive processed manifests."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/bdd100k_local.yaml",
        help="Path to dataset config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resolver = DatasetPathResolver(args.config)

    print("[dataset summary]")
    for k, v in resolver.summary().items():
        print(f"  {k}: {v}")

    train_df = build_absolute_manifest_from_portable(
        portable_manifest_path=resolver.train_portable_manifest,
        output_manifest_path=resolver.train_absolute_manifest,
        image_root=resolver.train_image_root,
        processed_masks_root=resolver.processed_masks_root,
        split="train",
    )

    val_df = build_absolute_manifest_from_portable(
        portable_manifest_path=resolver.val_portable_manifest,
        output_manifest_path=resolver.val_absolute_manifest,
        image_root=resolver.val_image_root,
        processed_masks_root=resolver.processed_masks_root,
        split="val",
    )

    print("\n[done]")
    print(f"  train rows: {len(train_df)}")
    print(f"  val rows:   {len(val_df)}")
    print(f"  train manifest: {resolver.train_absolute_manifest}")
    print(f"  val manifest:   {resolver.val_absolute_manifest}")


if __name__ == "__main__":
    main()