from __future__ import annotations

import json
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import torch


EPS = 1e-12


def set_seed(seed: int, num_threads: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if num_threads and num_threads > 0:
        torch.set_num_threads(int(num_threads))
        try:
            torch.set_num_interop_threads(max(1, int(num_threads)))
        except RuntimeError:
            pass


@dataclass
class ParameterPacker:
    names: List[str]
    shapes: List[torch.Size]
    sizes: List[int]
    total_size: int
    name_to_idx: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "ParameterPacker":
        names: List[str] = []
        shapes: List[torch.Size] = []
        sizes: List[int] = []
        total = 0
        for name, param in model.named_parameters():
            names.append(name)
            shapes.append(param.shape)
            numel = int(param.numel())
            sizes.append(numel)
            total += numel
        return cls(
            names=names,
            shapes=shapes,
            sizes=sizes,
            total_size=total,
            name_to_idx={name: i for i, name in enumerate(names)},
        )

    def flatten_state_dict(self, state_dict: Dict[str, torch.Tensor], device: torch.device | None = None) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for name in self.names:
            tensor = state_dict[name].detach().float().reshape(-1)
            if device is not None:
                tensor = tensor.to(device)
            chunks.append(tensor)
        if not chunks:
            return torch.empty(0, device=device or "cpu")
        return torch.cat(chunks)

    def flatten_model(self, model: torch.nn.Module, device: torch.device | None = None) -> torch.Tensor:
        return self.flatten_state_dict(model.state_dict(), device=device)

    def vector_to_state_dict(
        self,
        vector: torch.Tensor,
        reference_state: Dict[str, torch.Tensor],
        device: torch.device | None = None,
    ) -> "OrderedDict[str, torch.Tensor]":
        new_state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        ptr = 0
        for key, value in reference_state.items():
            if key in self.name_to_idx:
                idx = self.name_to_idx[key]
                numel = self.sizes[idx]
                chunk = vector[ptr:ptr + numel].reshape(self.shapes[idx]).to(dtype=value.dtype)
                if device is not None:
                    chunk = chunk.to(device)
                else:
                    chunk = chunk.to(value.device)
                new_state[key] = chunk
                ptr += numel
            else:
                new_state[key] = value.clone()
        return new_state


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a_norm = torch.linalg.norm(a)
    b_norm = torch.linalg.norm(b)
    if a_norm.item() < EPS or b_norm.item() < EPS:
        return 0.0
    return float(torch.dot(a, b) / (a_norm * b_norm + EPS))


@dataclass
class SparseUpdate:
    indices: torch.Tensor
    values: torch.Tensor
    total_size: int
    epsilon_t: float
    sigma_t: float
    communication_bytes: int
    sparse: bool = True

    def to_dense(self, device: torch.device | None = None) -> torch.Tensor:
        target_device = device or self.values.device
        if not self.sparse:
            return self.values.to(target_device)
        dense = torch.zeros(self.total_size, dtype=self.values.dtype, device=target_device)
        if self.indices.numel() > 0:
            dense[self.indices.long()] = self.values.to(dense.device)
        return dense


@dataclass
class RoundMetrics:
    round_idx: int
    train_loss: float
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    test_loss: float = 0.0
    test_accuracy: float = 0.0
    last10_mean_accuracy: float = 0.0
    mean_local_cosine: float = 0.0
    kept_batch_ratio: float = 0.0
    mean_server_cosine: float = 0.0
    mean_epsilon_t: float = 0.0
    mean_sigma_t: float = 0.0
    communication_mb: float = 0.0
    mean_clean_update_norm: float = 0.0
    mean_noisy_update_norm: float = 0.0



def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def compute_round_epsilon(
    epsilon: float,
    round_idx: int,
    total_rounds: int,
    mode: str = "practical",
) -> float:
    t = round_idx + 1
    denom = total_rounds * (total_rounds + 1) / 2.0
    if mode == "exact":
        return float(epsilon * t / denom)
    if mode == "practical":
        mean_round = (total_rounds + 1) / 2.0
        return float(epsilon * t / mean_round)
    raise ValueError(f"Unsupported dp_mode: {mode}")



def compute_sigma_from_budget(
    epsilon_t: float,
    delta: float,
    sigma_max: float = 5.0,
) -> float:
    epsilon_t = max(epsilon_t, 1e-6)
    delta = min(max(delta, 1e-12), 0.999999)
    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon_t
    return float(min(sigma, sigma_max))



def topk_indices(values: torch.Tensor, ratio: float) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    if ratio >= 1.0:
        return torch.arange(values.numel(), device=values.device)
    k = max(1, int(round(values.numel() * ratio)))
    k = min(k, values.numel())
    _, idx = torch.topk(values.abs(), k=k, largest=True, sorted=False)
    return idx.sort().values



def estimate_communication_bytes(num_values: int, sparse: bool) -> int:
    if num_values <= 0:
        return 0
    if sparse:
        return int(num_values * 8)  # int32 index + float32 value
    return int(num_values * 4)  # dense float32 payload
