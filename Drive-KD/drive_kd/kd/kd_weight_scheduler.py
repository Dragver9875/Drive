from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KDWeights:

    supervised: float = 1.0
    probability_kd: float = 0.5
    boundary_kd: float = 0.3
    attention_kd: float = 0.1
    feature_kd: float = 0.0


class KDWeightScheduler:

    def __init__(
        self,
        *,
        supervised: float = 1.0,
        probability_kd: float = 0.5,
        boundary_kd: float = 0.3,
        attention_kd: float = 0.1,
        feature_kd: float = 0.0,
        enabled: bool = True,
        kd_warmup_epochs: int = 3,
        mode: str = "linear",
    ) -> None:
        self.base = KDWeights(
            supervised=float(supervised),
            probability_kd=float(probability_kd),
            boundary_kd=float(boundary_kd),
            attention_kd=float(attention_kd),
            feature_kd=float(feature_kd),
        )

        self.enabled = bool(enabled)
        self.kd_warmup_epochs = int(kd_warmup_epochs)
        self.mode = str(mode).lower()

        if self.kd_warmup_epochs < 0:
            raise ValueError("kd_warmup_epochs must be >= 0")

        if self.mode not in {"linear", "constant"}:
            raise ValueError(f"Unsupported KD scheduler mode: {self.mode}")

    @classmethod
    def from_config(cls, cfg: dict) -> "KDWeightScheduler":
        kd_cfg = cfg.get("kd", cfg)
        weights_cfg = kd_cfg.get("weights", {})
        sched_cfg = kd_cfg.get("scheduling", {})

        return cls(
            supervised=float(weights_cfg.get("supervised", 1.0)),
            probability_kd=float(weights_cfg.get("probability_kd", 0.5)),
            boundary_kd=float(weights_cfg.get("boundary_kd", 0.3)),
            attention_kd=float(weights_cfg.get("attention_kd", 0.1)),
            feature_kd=float(weights_cfg.get("feature_kd", 0.0)),
            enabled=bool(sched_cfg.get("enabled", True)),
            kd_warmup_epochs=int(sched_cfg.get("kd_warmup_epochs", 3)),
            mode=str(sched_cfg.get("mode", "linear")),
        )

    def kd_scale(self, epoch: int) -> float:
        if not self.enabled:
            return 1.0

        if self.mode == "constant":
            return 1.0

        if self.kd_warmup_epochs == 0:
            return 1.0

        epoch = max(1, int(epoch))

        return min(1.0, float(epoch) / float(self.kd_warmup_epochs))

    def get_weights(self, epoch: int) -> KDWeights:
        scale = self.kd_scale(epoch)

        return KDWeights(
            supervised=self.base.supervised,
            probability_kd=self.base.probability_kd * scale,
            boundary_kd=self.base.boundary_kd * scale,
            attention_kd=self.base.attention_kd * scale,
            feature_kd=self.base.feature_kd * scale,
        )

    def get_weights_dict(self, epoch: int) -> dict[str, float]:
        w = self.get_weights(epoch)

        return {
            "supervised": w.supervised,
            "probability_kd": w.probability_kd,
            "boundary_kd": w.boundary_kd,
            "attention_kd": w.attention_kd,
            "feature_kd": w.feature_kd,
        }