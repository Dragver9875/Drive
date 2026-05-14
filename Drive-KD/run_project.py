from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


DEFAULTS = {
    "dataset_config": "configs/dataset/bdd100k_local.yaml",
    "teacher_exp": "configs/experiments/exp_000_teacher_segformer_b1.yaml",
    "cache_exp": "configs/experiments/exp_001_teacher_cache.yaml",
    "student_exp": "configs/experiments/exp_002_student_effb0_full_kd.yaml",
    "teacher_checkpoint": "outputs/teacher_segformer_b1/checkpoints/best.pt",
    "student_checkpoint": "outputs/student_effb0_kd/checkpoints/best.pt",
}


VALID_STEPS = [
    "tests",
    "manifests",
    "check",
    "teacher",
    "cache",
    "student",
    "evaluate",
    "benchmark",
    "export",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Drive-KD Track C pipeline: "
            "manifests -> dataset check -> teacher -> cache -> student -> eval -> benchmark -> export."
        )
    )

    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help=(
            "Comma-separated steps to run. "
            f"Available: all,{','.join(VALID_STEPS)}"
        ),
    )

    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use.",
    )

    parser.add_argument(
        "--dataset-config",
        type=str,
        default=DEFAULTS["dataset_config"],
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
        "--teacher-checkpoint",
        type=str,
        default=DEFAULTS["teacher_checkpoint"],
    )

    parser.add_argument(
        "--student-checkpoint",
        type=str,
        default=DEFAULTS["student_checkpoint"],
    )

    parser.add_argument(
        "--benchmark-batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--export-format",
        type=str,
        default="all",
        choices=["pt", "torchscript", "onnx", "all"],
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest step even if steps=all.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later steps if a step fails.",
    )

    return parser.parse_args()


def normalize_steps(steps_arg: str, skip_tests: bool) -> list[str]:
    if steps_arg.strip().lower() == "all":
        steps = VALID_STEPS.copy()
    else:
        steps = [s.strip().lower() for s in steps_arg.split(",") if s.strip()]

    invalid = [s for s in steps if s not in VALID_STEPS]

    if invalid:
        raise ValueError(f"Invalid steps: {invalid}. Valid steps: {VALID_STEPS}")

    if skip_tests and "tests" in steps:
        steps.remove("tests")

    return steps


def run_command(
    *,
    name: str,
    cmd: list[str],
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> bool:
    print("\n" + "=" * 88)
    print(f"[Drive-KD] STEP: {name}")
    print("=" * 88)
    print(" ".join(cmd))

    if dry_run:
        return True

    start = time.time()

    try:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - start
        print(f"[FAILED] {name} after {elapsed / 60.0:.2f} min")
        print(f"Return code: {exc.returncode}")

        if continue_on_error:
            return False

        raise

    elapsed = time.time() - start
    print(f"[OK] {name} completed in {elapsed / 60.0:.2f} min")
    return True


def build_commands(args: argparse.Namespace) -> dict[str, tuple[str, list[str]]]:
    py = args.python

    return {
        "tests": (
            "Run unit tests",
            [
                py,
                "-m",
                "pytest",
                "-q",
            ],
        ),
        "manifests": (
            "Build absolute manifests",
            [
                py,
                "scripts/00_build_absolute_manifests.py",
                "--config",
                args.dataset_config,
            ],
        ),
        "check": (
            "Check dataset integrity",
            [
                py,
                "scripts/01_check_dataset.py",
                "--config",
                args.dataset_config,
            ],
        ),
        "teacher": (
            "Train SegFormer-B1 teacher",
            [
                py,
                "scripts/02_train_teacher_segformer_b1.py",
                "--config",
                args.teacher_exp,
            ],
        ),
        "cache": (
            "Generate teacher cache",
            [
                py,
                "scripts/03_generate_teacher_cache.py",
                "--config",
                args.cache_exp,
                "--checkpoint",
                args.teacher_checkpoint,
            ],
        ),
        "student": (
            "Train Drive-EffB0-BiFPN-KD student",
            [
                py,
                "scripts/04_train_student_effb0_kd.py",
                "--config",
                args.student_exp,
            ],
        ),
        "evaluate": (
            "Evaluate KD student",
            [
                py,
                "scripts/05_evaluate_student.py",
                "--config",
                args.student_exp,
                "--checkpoint",
                args.student_checkpoint,
            ],
        ),
        "benchmark": (
            "Benchmark KD student FPS",
            [
                py,
                "scripts/06_benchmark_fps.py",
                "--config",
                args.student_exp,
                "--checkpoint",
                args.student_checkpoint,
                "--batch-size",
                str(args.benchmark_batch_size),
                "--image-height",
                str(args.image_height),
                "--image-width",
                str(args.image_width),
            ],
        ),
        "export": (
            "Export KD student",
            [
                py,
                "scripts/07_export_student.py",
                "--config",
                args.student_exp,
                "--checkpoint",
                args.student_checkpoint,
                "--format",
                args.export_format,
                "--image-height",
                str(args.image_height),
                "--image-width",
                str(args.image_width),
            ],
        ),
    }


def check_repo_sanity() -> None:
    required_paths = [
        "drive_kd",
        "configs",
        "scripts",
        "requirements.txt",
        "pyproject.toml",
        "scripts/00_build_absolute_manifests.py",
        "scripts/01_check_dataset.py",
        "scripts/02_train_teacher_segformer_b1.py",
        "scripts/03_generate_teacher_cache.py",
        "scripts/04_train_student_effb0_kd.py",
        "scripts/05_evaluate_student.py",
        "scripts/06_benchmark_fps.py",
        "scripts/07_export_student.py",
    ]

    missing = []

    for rel in required_paths:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)

    if missing:
        raise FileNotFoundError(
            "Repo sanity check failed. Missing paths:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def main() -> None:
    args = parse_args()
    check_repo_sanity()

    steps = normalize_steps(args.steps, skip_tests=args.skip_tests)
    commands = build_commands(args)

    print("[Drive-KD] Project root:", REPO_ROOT)
    print("[Drive-KD] Selected steps:", ", ".join(steps))

    overall_start = time.time()
    results: dict[str, bool] = {}

    for step in steps:
        name, cmd = commands[step]

        ok = run_command(
            name=name,
            cmd=cmd,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        results[step] = ok

    elapsed = time.time() - overall_start

    print("\n" + "=" * 88)
    print("[Drive-KD] RUN SUMMARY")
    print("=" * 88)

    for step in steps:
        status = "OK" if results.get(step, False) else "FAILED"
        print(f"{step:>12}: {status}")

    print(f"\nTotal elapsed: {elapsed / 60.0:.2f} min")

    failed = [step for step, ok in results.items() if not ok]

    if failed:
        raise SystemExit(f"Failed steps: {failed}")

    print("[Drive-KD] All selected steps completed.")


if __name__ == "__main__":
    main()