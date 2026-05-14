from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drive_kd.datasets import DriveSupervisedDataset, drive_supervised_collate
from drive_kd.engine import (
    load_checkpoint,
    make_grad_scaler,
    save_checkpoint,
    train_teacher_one_epoch,
    validate_one_epoch,
)
from drive_kd.engine.checkpointing import checkpoint_metric_is_better
from drive_kd.losses import SupervisedMultiTaskLoss
from drive_kd.metrics import DriveEvaluator
from drive_kd.models.teachers import build_teacher
from drive_kd.utils import (
    get_amp_enabled,
    get_device,
    load_config,
    save_config_snapshot,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer-B1 teacher.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp_000_teacher_segformer_b1.yaml",
    )
    return parser.parse_args()


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": drive_supervised_collate,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, include_linked_configs=True)

    dataset_cfg = cfg["resolved"]["dataset"]["dataset"]
    teacher_cfg = cfg["resolved"]["teacher"]

    exp = cfg["experiment"]
    training = cfg["training"]
    output = cfg["output"]
    checkpointing = cfg["checkpointing"]
    metrics_cfg = cfg["metrics"]

    seed_everything(int(exp.get("seed", 42)))

    device = get_device()
    amp_enabled = get_amp_enabled(device, bool(training.get("amp", True)))

    run_dir = Path(output["run_dir"])
    ckpt_dir = Path(output["checkpoint_dir"])
    metrics_dir = Path(output["metrics_dir"])
    config_dir = Path(output["config_dir"])

    for p in [run_dir, ckpt_dir, metrics_dir, config_dir, Path(output["visualization_dir"])]:
        p.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(cfg, config_dir, "resolved_experiment_config.json")

    train_ds = DriveSupervisedDataset(
        manifest_path=dataset_cfg["absolute_manifests"]["train"],
        image_height=int(dataset_cfg["image_height"]),
        image_width=int(dataset_cfg["image_width"]),
        mean=dataset_cfg["normalization"]["mean"],
        std=dataset_cfg["normalization"]["std"],
        train=True,
        transform_config=dataset_cfg.get("transforms", {}),
    )

    val_ds = DriveSupervisedDataset(
        manifest_path=dataset_cfg["absolute_manifests"]["val"],
        image_height=int(dataset_cfg["image_height"]),
        image_width=int(dataset_cfg["image_width"]),
        mean=dataset_cfg["normalization"]["mean"],
        std=dataset_cfg["normalization"]["std"],
        train=False,
        transform_config=dataset_cfg.get("transforms", {}),
    )

    train_loader = make_loader(
        train_ds,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training.get("num_workers", 2)),
    )

    val_loader = make_loader(
        val_ds,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training.get("num_workers", 2)),
    )

    model = build_teacher(teacher_cfg).to(device)

    optimizer_cfg = training["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
    )

    scheduler_cfg = training["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["epochs"]),
        eta_min=float(scheduler_cfg.get("min_lr", 1e-6)),
    )

    scaler = make_grad_scaler(enabled=amp_enabled, device=device)
    criterion = SupervisedMultiTaskLoss.from_config(cfg).to(device)
    evaluator = DriveEvaluator.from_config(cfg)

    start_epoch = 1
    best_metric = None
    history: list[dict] = []

    last_ckpt = ckpt_dir / "last.pt"

    if bool(training.get("resume", True)) and last_ckpt.exists():
        ckpt = load_checkpoint(
            path=last_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
            strict=True,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = ckpt.get("best_metric")
        history = ckpt.get("history", [])
        print(f"[resume] {last_ckpt}, start_epoch={start_epoch}, best_metric={best_metric}")

    monitor = str(checkpointing.get("monitor", metrics_cfg.get("monitor", "val/lane_f1")))
    mode = str(checkpointing.get("mode", metrics_cfg.get("mode", "max")))

    start_time = time.time()

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        train_log = train_teacher_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            grad_accum_steps=int(training.get("grad_accum_steps", 1)),
            amp_enabled=amp_enabled,
            max_grad_norm=float(training.get("gradient_clipping", {}).get("max_norm", 5.0)),
        )

        val_log = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            evaluator=evaluator,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
            desc=f"teacher val epoch {epoch}",
        )

        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_log,
            **val_log,
        }

        history.append(epoch_log)
        print(json.dumps(epoch_log, indent=2))

        current_metric = float(epoch_log.get(monitor, 0.0))

        save_checkpoint(
            path=ckpt_dir / "last.pt",
            model=model,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=cfg,
            metrics=epoch_log,
            history=history,
            best_metric=best_metric,
        )

        if bool(checkpointing.get("save_every_epoch", True)):
            save_checkpoint(
                path=ckpt_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=cfg,
                metrics=epoch_log,
                history=history,
                best_metric=best_metric,
            )

        if checkpoint_metric_is_better(current_metric, best_metric, mode=mode):
            best_metric = current_metric
            save_checkpoint(
                path=ckpt_dir / "best.pt",
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=cfg,
                metrics=epoch_log,
                history=history,
                best_metric=best_metric,
            )
            print(f"[best] {monitor}={best_metric:.6f}")

        write_json(history, metrics_dir / "history.json")

        best_metrics = {
            "monitor": monitor,
            "mode": mode,
            "best_metric": best_metric,
            "latest_epoch": epoch,
            "latest_metrics": epoch_log,
            "elapsed_minutes": (time.time() - start_time) / 60.0,
        }
        write_json(best_metrics, metrics_dir / "best_metrics.json")

    print("[done] Teacher training complete.")


if __name__ == "__main__":
    main()