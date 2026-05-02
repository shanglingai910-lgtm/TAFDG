from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional
import json
import os

import torch


@dataclass
class TAFDGConfig:
    dataset: str = "synthetic"  # synthetic | imagefolder | gtsrb | tt100k | miotcd
    data_root: str = ""
    domains: Optional[List[str]] = None
    holdout_domain: Optional[str] = None
    holdout_index: int = 0

    output_dir: str = "outputs/tafdg"
    image_size: int = 64
    num_classes: int = 10
    num_domains: int = 4
    num_clients: int = 20
    clients_per_round: float = 0.2
    dirichlet_alpha: float = 1.0

    rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9

    model: str = "tinycnn"
    disable_local_align: bool = False
    disable_server_align: bool = False
    disable_topk: bool = False
    disable_dp: bool = False

    tau: float = 0.1
    align_warmup_rounds: int = 1
    method_warmup_rounds: int = 0  # engineering warmup: plain FedAvg rounds before enabling full TAFDG
    min_kept_ratio: float = 0.25
    min_kept_clients: int = 1
    align_rescue_scale: float = 0.35
    reference_blend: float = 0.5
    reference_reset_cosine: float = -0.05
    topk_ratio: float = 0.1
    clip_norm: float = 1.0
    epsilon: float = 5.0
    delta: float = 1e-5
    dp_mode: str = "exact"  # exact follows the paper Eq. (15); practical is a looser engineering mode
    sigma_max: float = 5.0
    dp_noise_scale_mode: str = "l2_normalized"  # per_coordinate | l2_normalized

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    num_threads: int = 0  # 0 means keep PyTorch default thread setting
    random_hflip: bool = False

    # Dataset download/cache controls
    auto_download: bool = True
    dataset_cache_dir: str = "datasets"
    preset_use_official_test: bool = True
    val_ratio: float = 0.1

    # Synthetic-only controls
    synthetic_samples_per_class: int = 64
    synthetic_noise: float = 0.08
    synthetic_style_strength: float = 0.35

    # Classification-preset controls
    preset_split_mode: str = "balanced"  # balanced | sequential
    min_class_total_samples: int = 2  # classes below this are dropped from proxy-domain benchmarks

    # Evaluation and checkpoints
    checkpoint_every: int = 0
    save_last: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
