from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent


DEFAULTS = {
    "dataset_config": "configs/dataset/bdd100k_local.yaml",
    "teacher_config": "configs/teacher/segformer_b1_teacher.yaml",
    "student_config": "configs/student/drive_effb0_bifpn_kd.yaml",
    "kd_config": "configs/kd/kd_full.yaml",
    "teacher_exp": "configs/experiments/exp_000_teacher_segformer_b1.yaml",
    "cache_exp": "configs/experiments/exp_001_teacher_cache.yaml",
    "student_exp": "configs/experiments/exp_002_student_effb0_full_kd.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny Drive-KD smoke pipeline using temporary configs and subset manifests. "
            "This avoids full teacher-cache generation over the complete BDD100K dataset."
        )
    )

    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
    )

    parser.add_argument(
        "--dataset-config",
        type=str,
        default=DEFAULTS["dataset_config"],
    )

    parser.add_argument(
        "--teacher-config",
        type=str,
        default=DEFAULTS["teacher_config"],
    )

    parser.add_argument(
        "--student-config",
        type=str,
        default=DEFAULTS["student_config"],
    )

    parser.add_argument(
        "--kd-config",
        type=str,
        default=DEFAULTS["kd_config"],
    )

    parser.add_argument(
        "--teacher-exp",
        type=str,
        default=DEFAULTS["teacher_exp"],
    )

    parser.add_argument(
        "--cache-exp",
        type=str,
        default=DEFAULTS["cache_exp"],
    )

    parser.add_argument(
        "--student-exp",
        type=str,
        default=DEFAULTS["student_exp"],
    )

    parser.add_argument(
        "--smoke-root",
        type=str,
        default="outputs/smoke",
        help="Directory for smoke outputs and smoke manifests.",
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=32,
        help="Number of training rows to use in smoke manifests.",
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=8,
        help="Number of validation rows to use in smoke manifests.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--student-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--offline-model-init",
        action="store_true",
        help=(
            "Use pretrained=False for teacher and student smoke configs. "
            "This avoids downloading Hugging Face/timm weights. "
            "Use only for code-flow smoke testing, not quality."
        ),
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
    )

    parser.add_argument(
        "--skip-manifest-build",
        action="store_true",
        help="Reuse existing data/manifests/train_absolute.csv and val_absolute.csv.",
    )

    parser.add_argument(
        "--include-benchmark",
        action="store_true",
    )

    parser.add_argument(
        "--include-export",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict from YAML: {path}")

    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    return path


def rel(path: str | Path) -> str:
    """
    Return repo-relative POSIX path when possible.
    """

    p = Path(path)

    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def run_command(
    *,
    name: str,
    cmd: list[str],
    dry_run: bool = False,
) -> None:
    print("\n" + "=" * 88)
    print(f"[Drive-KD Smoke] STEP: {name}")
    print("=" * 88)
    print(" ".join(cmd))

    if dry_run:
        return

    start = time.time()

    subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
    )

    elapsed = time.time() - start
    print(f"[OK] {name} completed in {elapsed / 60.0:.2f} min")


def update_output_paths_for_teacher_exp(
    cfg: dict[str, Any],
    smoke_root: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    run_dir = smoke_root / "teacher_segformer_b1"

    cfg["output"] = {
        "run_dir": rel(run_dir),
        "checkpoint_dir": rel(run_dir / "checkpoints"),
        "metrics_dir": rel(run_dir / "metrics"),
        "visualization_dir": rel(run_dir / "visualizations"),
        "config_dir": rel(run_dir / "configs"),
    }

    return cfg


def update_output_paths_for_student_exp(
    cfg: dict[str, Any],
    smoke_root: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    run_dir = smoke_root / "student_effb0_kd"

    cfg["output"] = {
        "run_dir": rel(run_dir),
        "checkpoint_dir": rel(run_dir / "checkpoints"),
        "metrics_dir": rel(run_dir / "metrics"),
        "visualization_dir": rel(run_dir / "visualizations"),
        "config_dir": rel(run_dir / "configs"),
    }

    return cfg


def create_subset_manifest(
    *,
    input_manifest: Path,
    output_manifest: Path,
    n: int,
    prefer_positive_lane: bool = True,
) -> pd.DataFrame:
    if not input_manifest.exists():
        raise FileNotFoundError(
            f"Absolute manifest missing: {input_manifest}\n"
            "Run manifest build first."
        )

    df = pd.read_csv(input_manifest)

    if len(df) == 0:
        raise RuntimeError(f"Input manifest has zero rows: {input_manifest}")

    n = min(int(n), len(df))

    if prefer_positive_lane and "has_lane" in df.columns:
        lane_df = df[df["has_lane"].fillna(0).astype(int) == 1]
        other_df = df[df["has_lane"].fillna(0).astype(int) != 1]

        selected = pd.concat([lane_df, other_df], axis=0).head(n).copy()
    else:
        selected = df.head(n).copy()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_manifest, index=False)

    return selected


def prepare_smoke_configs(args: argparse.Namespace) -> dict[str, Path]:
    smoke_root = (REPO_ROOT / args.smoke_root).resolve()
    smoke_config_dir = REPO_ROOT / "configs" / "smoke"
    smoke_data_dir = smoke_root / "data"
    smoke_manifest_dir = smoke_data_dir / "manifests"
    smoke_cache_root = smoke_data_dir / "cache" / "teacher_segformer_b1"

    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke_config_dir.mkdir(parents=True, exist_ok=True)
    smoke_manifest_dir.mkdir(parents=True, exist_ok=True)

    base_dataset_cfg_path = REPO_ROOT / args.dataset_config
    base_teacher_cfg_path = REPO_ROOT / args.teacher_config
    base_student_cfg_path = REPO_ROOT / args.student_config
    base_kd_cfg_path = REPO_ROOT / args.kd_config
    base_teacher_exp_path = REPO_ROOT / args.teacher_exp
    base_cache_exp_path = REPO_ROOT / args.cache_exp
    base_student_exp_path = REPO_ROOT / args.student_exp

    dataset_cfg = load_yaml(base_dataset_cfg_path)
    teacher_cfg = load_yaml(base_teacher_cfg_path)
    student_cfg = load_yaml(base_student_cfg_path)
    kd_cfg = load_yaml(base_kd_cfg_path)

    teacher_exp = load_yaml(base_teacher_exp_path)
    cache_exp = load_yaml(base_cache_exp_path)
    student_exp = load_yaml(base_student_exp_path)

    full_train_manifest = REPO_ROOT / dataset_cfg["dataset"]["absolute_manifests"]["train"]
    full_val_manifest = REPO_ROOT / dataset_cfg["dataset"]["absolute_manifests"]["val"]

    smoke_train_abs = smoke_manifest_dir / "train_absolute_smoke.csv"
    smoke_val_abs = smoke_manifest_dir / "val_absolute_smoke.csv"
    smoke_train_kd = smoke_manifest_dir / "train_kd_smoke.csv"
    smoke_val_kd = smoke_manifest_dir / "val_kd_smoke.csv"

    train_subset = create_subset_manifest(
        input_manifest=full_train_manifest,
        output_manifest=smoke_train_abs,
        n=args.train_samples,
        prefer_positive_lane=True,
    )

    val_subset = create_subset_manifest(
        input_manifest=full_val_manifest,
        output_manifest=smoke_val_abs,
        n=args.val_samples,
        prefer_positive_lane=True,
    )

    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg["dataset"]["absolute_manifests"]["train"] = rel(smoke_train_abs)
    dataset_cfg["dataset"]["absolute_manifests"]["val"] = rel(smoke_val_abs)
    dataset_cfg["dataset"]["kd_manifests"]["train"] = rel(smoke_train_kd)
    dataset_cfg["dataset"]["kd_manifests"]["val"] = rel(smoke_val_kd)

    dataset_cfg["dataset"]["expected_counts"] = {
        "train_min_rows": 1,
        "val_min_rows": 1,
        "train_expected_rows": int(len(train_subset)),
        "val_expected_rows": int(len(val_subset)),
    }

    dataset_cfg["dataset"]["loader"]["num_workers"] = int(args.num_workers)

    smoke_dataset_cfg_path = smoke_config_dir / "bdd100k_local_smoke.yaml"
    save_yaml(dataset_cfg, smoke_dataset_cfg_path)

    teacher_cfg = copy.deepcopy(teacher_cfg)

    if args.offline_model_init:
        teacher_cfg["teacher"]["pretrained"]["enabled"] = False

    smoke_teacher_cfg_path = smoke_config_dir / "segformer_b1_teacher_smoke.yaml"
    save_yaml(teacher_cfg, smoke_teacher_cfg_path)

    student_cfg = copy.deepcopy(student_cfg)

    if args.offline_model_init:
        student_cfg["student"]["encoder"]["pretrained"] = False

    smoke_student_cfg_path = smoke_config_dir / "drive_effb0_bifpn_kd_smoke.yaml"
    save_yaml(student_cfg, smoke_student_cfg_path)

    kd_cfg = copy.deepcopy(kd_cfg)

    kd_cfg["kd"]["teacher"]["cache_root"] = rel(smoke_cache_root)
    kd_cfg["kd"]["teacher"]["probabilities"]["train_dir"] = rel(smoke_cache_root / "probabilities" / "train")
    kd_cfg["kd"]["teacher"]["probabilities"]["val_dir"] = rel(smoke_cache_root / "probabilities" / "val")
    kd_cfg["kd"]["teacher"]["boundaries"]["train_dir"] = rel(smoke_cache_root / "boundaries" / "train")
    kd_cfg["kd"]["teacher"]["boundaries"]["val_dir"] = rel(smoke_cache_root / "boundaries" / "val")
    kd_cfg["kd"]["teacher"]["attention"]["train_dir"] = rel(smoke_cache_root / "attention" / "train")
    kd_cfg["kd"]["teacher"]["attention"]["val_dir"] = rel(smoke_cache_root / "attention" / "val")

    smoke_kd_cfg_path = smoke_config_dir / "kd_full_smoke.yaml"
    save_yaml(kd_cfg, smoke_kd_cfg_path)

    teacher_exp = update_output_paths_for_teacher_exp(teacher_exp, smoke_root)
    teacher_exp["experiment"]["name"] = "smoke_teacher_segformer_b1"
    teacher_exp["experiment"]["dataset_config"] = rel(smoke_dataset_cfg_path)
    teacher_exp["experiment"]["teacher_config"] = rel(smoke_teacher_cfg_path)

    teacher_exp["training"]["epochs"] = int(args.teacher_epochs)
    teacher_exp["training"]["batch_size"] = int(args.batch_size)
    teacher_exp["training"]["grad_accum_steps"] = 1
    teacher_exp["training"]["num_workers"] = int(args.num_workers)
    teacher_exp["training"]["resume"] = False

    teacher_exp["checkpointing"]["save_every_epoch"] = False
    teacher_exp["validation"]["max_visualizations"] = min(2, int(args.val_samples))

    smoke_teacher_exp_path = smoke_config_dir / "exp_000_teacher_smoke.yaml"
    save_yaml(teacher_exp, smoke_teacher_exp_path)

    cache_exp = copy.deepcopy(cache_exp)

    smoke_teacher_best = smoke_root / "teacher_segformer_b1" / "checkpoints" / "best.pt"

    cache_exp["experiment"]["name"] = "smoke_teacher_cache"
    cache_exp["experiment"]["dataset_config"] = rel(smoke_dataset_cfg_path)
    cache_exp["experiment"]["teacher_config"] = rel(smoke_teacher_cfg_path)
    cache_exp["experiment"]["cache_config"] = "configs/teacher/segformer_b1_cache.yaml"

    cache_exp["teacher"]["checkpoint"] = rel(smoke_teacher_best)
    cache_exp["output"]["cache_root"] = rel(smoke_cache_root)
    cache_exp["output"]["reports_dir"] = rel(smoke_cache_root / "reports")

    cache_exp["cache"]["splits"] = ["train", "val"]
    cache_exp["cache"]["batch_size"] = int(args.batch_size)
    cache_exp["cache"]["num_workers"] = int(args.num_workers)
    cache_exp["cache"]["overwrite_existing"] = True
    cache_exp["cache"]["verify_after_write"] = True

    cache_exp["kd_manifest"]["train_output"] = rel(smoke_train_kd)
    cache_exp["kd_manifest"]["val_output"] = rel(smoke_val_kd)

    smoke_cache_exp_path = smoke_config_dir / "exp_001_teacher_cache_smoke.yaml"
    save_yaml(cache_exp, smoke_cache_exp_path)

    student_exp = update_output_paths_for_student_exp(student_exp, smoke_root)
    student_exp["experiment"]["name"] = "smoke_student_effb0_full_kd"
    student_exp["experiment"]["dataset_config"] = rel(smoke_dataset_cfg_path)
    student_exp["experiment"]["student_config"] = rel(smoke_student_cfg_path)
    student_exp["experiment"]["kd_config"] = rel(smoke_kd_cfg_path)

    student_exp["data"]["train_manifest"] = rel(smoke_train_kd)
    student_exp["data"]["val_manifest"] = rel(smoke_val_kd)

    student_exp["training"]["epochs"] = int(args.student_epochs)
    student_exp["training"]["batch_size"] = int(args.batch_size)
    student_exp["training"]["grad_accum_steps"] = 1
    student_exp["training"]["num_workers"] = int(args.num_workers)
    student_exp["training"]["resume"] = False
    student_exp["training"]["init_from_supervised_student"]["enabled"] = False

    student_exp["checkpointing"]["save_every_epoch"] = False
    student_exp["validation"]["max_visualizations"] = min(2, int(args.val_samples))

    smoke_student_exp_path = smoke_config_dir / "exp_002_student_effb0_full_kd_smoke.yaml"
    save_yaml(student_exp, smoke_student_exp_path)

    print("[smoke configs created]")
    print(f"  smoke root:        {smoke_root}")
    print(f"  train rows:        {len(train_subset)}")
    print(f"  val rows:          {len(val_subset)}")
    print(f"  dataset config:    {smoke_dataset_cfg_path}")
    print(f"  teacher config:    {smoke_teacher_cfg_path}")
    print(f"  student config:    {smoke_student_cfg_path}")
    print(f"  kd config:         {smoke_kd_cfg_path}")
    print(f"  teacher exp:       {smoke_teacher_exp_path}")
    print(f"  cache exp:         {smoke_cache_exp_path}")
    print(f"  student exp:       {smoke_student_exp_path}")

    return {
        "smoke_root": smoke_root,
        "dataset_config": smoke_dataset_cfg_path,
        "teacher_config": smoke_teacher_cfg_path,
        "student_config": smoke_student_cfg_path,
        "kd_config": smoke_kd_cfg_path,
        "teacher_exp": smoke_teacher_exp_path,
        "cache_exp": smoke_cache_exp_path,
        "student_exp": smoke_student_exp_path,
        "teacher_checkpoint": smoke_teacher_best,
        "student_checkpoint": smoke_root / "student_effb0_kd" / "checkpoints" / "best.pt",
    }


def main() -> None:
    args = parse_args()

    print("[Drive-KD Smoke] Project root:", REPO_ROOT)

    py = args.python

    start = time.time()

    if not args.skip_tests:
        run_command(
            name="Run lightweight tests",
            cmd=[
                py,
                "-m",
                "pytest",
                "-q",
                "tests/test_imports.py",
                "tests/test_student_forward.py",
                "tests/test_losses.py",
                "tests/test_kd_losses.py",
                "tests/test_metrics.py",
            ],
            dry_run=args.dry_run,
        )

    if not args.skip_manifest_build:
        run_command(
            name="Build full absolute manifests",
            cmd=[
                py,
                "scripts/00_build_absolute_manifests.py",
                "--config",
                args.dataset_config,
            ],
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("[dry-run] Skipping smoke config generation.")
        return

    paths = prepare_smoke_configs(args)

    run_command(
        name="Check smoke dataset",
        cmd=[
            py,
            "scripts/01_check_dataset.py",
            "--config",
            rel(paths["dataset_config"]),
            "--num-samples",
            str(min(args.train_samples, 32)),
        ],
        dry_run=False,
    )

    run_command(
        name="Train smoke teacher",
        cmd=[
            py,
            "scripts/02_train_teacher_segformer_b1.py",
            "--config",
            rel(paths["teacher_exp"]),
        ],
        dry_run=False,
    )

    run_command(
        name="Generate smoke teacher cache",
        cmd=[
            py,
            "scripts/03_generate_teacher_cache.py",
            "--config",
            rel(paths["cache_exp"]),
            "--checkpoint",
            rel(paths["teacher_checkpoint"]),
        ],
        dry_run=False,
    )

    run_command(
        name="Train smoke KD student",
        cmd=[
            py,
            "scripts/04_train_student_effb0_kd.py",
            "--config",
            rel(paths["student_exp"]),
        ],
        dry_run=False,
    )

    run_command(
        name="Evaluate smoke KD student",
        cmd=[
            py,
            "scripts/05_evaluate_student.py",
            "--config",
            rel(paths["student_exp"]),
            "--checkpoint",
            rel(paths["student_checkpoint"]),
        ],
        dry_run=False,
    )

    if args.include_benchmark:
        run_command(
            name="Benchmark smoke KD student",
            cmd=[
                py,
                "scripts/06_benchmark_fps.py",
                "--config",
                rel(paths["student_exp"]),
                "--checkpoint",
                rel(paths["student_checkpoint"]),
                "--batch-size",
                "1",
                "--image-height",
                "384",
                "--image-width",
                "640",
                "--warmup-iters",
                "3",
                "--benchmark-iters",
                "5",
            ],
            dry_run=False,
        )

    if args.include_export:
        run_command(
            name="Export smoke KD student",
            cmd=[
                py,
                "scripts/07_export_student.py",
                "--config",
                rel(paths["student_exp"]),
                "--checkpoint",
                rel(paths["student_checkpoint"]),
                "--format",
                "pt",
                "--image-height",
                "384",
                "--image-width",
                "640",
            ],
            dry_run=False,
        )

    elapsed = time.time() - start

    print("\n" + "=" * 88)
    print("[Drive-KD Smoke] COMPLETE")
    print("=" * 88)
    print(f"Smoke root: {paths['smoke_root']}")
    print(f"Teacher checkpoint: {paths['teacher_checkpoint']}")
    print(f"Student checkpoint: {paths['student_checkpoint']}")
    print(f"Elapsed: {elapsed / 60.0:.2f} min")


if __name__ == "__main__":
    main()