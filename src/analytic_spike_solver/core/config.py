from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from ..generation.bias import BiasSpikeConfig


ThetaPolicy = Literal["constant", "proportional_tau", "target_gain"]


@dataclass(frozen=True)
class TauThetaConfig:
    tau: float = 0.02
    theta: float = 1.0
    theta_policy: ThetaPolicy = "constant"
    theta_per_tau: float | None = None
    target_gain: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WeightInitConfig:
    mode: str = "he_signed_centered_safe"
    gain: float = 1.0
    jitter: float = 0.6
    max_abs_weight: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LayerConfig:
    n_pre: int
    n_post: int
    tau_theta: TauThetaConfig = field(default_factory=TauThetaConfig)
    weight_init: WeightInitConfig = field(default_factory=WeightInitConfig)
    bias: float | list[float] | None = None
    bias_config: BiasSpikeConfig = field(default_factory=BiasSpikeConfig)
    delay: float = 0.0
    name: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        return out


@dataclass(frozen=True)
class NetworkConfig:
    layer_sizes: list[int]
    duration: float
    tau_theta: TauThetaConfig = field(default_factory=TauThetaConfig)
    weight_init: WeightInitConfig = field(default_factory=WeightInitConfig)
    bias: float | list[float] | None = None
    bias_config: BiasSpikeConfig = field(default_factory=BiasSpikeConfig)
    seed: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)
