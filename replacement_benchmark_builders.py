from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from .config import TAFDGConfig
from .data import (
    PRESET_DATASETS,
    BasePresetSample,
    BasicImageTransform,
    ClientSubsetDataset,
    FederatedBenchmark,
    ImageFolderDomainDataset,
    StyledVisionDataset,
    _build_client_subsets_from_train_samples,
    _collect_classification_samples,
    _ensure_gtsrb_available,
    _ensure_miotcd_available,
    _ensure_tt100k_available,
    _find_gtsrb_training_root,
    _load_csv_label_samples,
    _load_tt100k_samples,
    _scan_imagefolder_domains,
)

Sample5 = Tuple[str, int, str, str, Optional[Tuple[float, float, float, float]]]
Sample3 = Tuple[str, int, str]


@dataclass
class DatasetAudit:
    class_names: List[str]
    source_domains: List[str]
    holdout_domain: str
    n_train: int
    n_val: int
    n_test: int


def _split_per_domain_per_class(
    samples: List[Sample5],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Sample5], List[Sample5]]:
    if val_ratio <= 0.0:
        return list(samples), []

    rng = np.random.default_rng(seed)
    grouped: Dict[Tuple[str, int], List[int]] = {}
    for idx, (_path, label, domain, _style, _bbox) in enumerate(samples):
        grouped.setdefault((str(domain), int(label)), []).append(idx)

    val_indices = set()
    for key, idxs in grouped.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if len(idxs) <= 1:
            continue
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        n_val = min(n_val, len(idxs) - 1)
        val_indices.update(idxs[:n_val])

    train_samples = [sample for i, sample in enumerate(samples) if i not in val_indices]
    val_samples = [sample for i, sample in enumerate(samples) if i in val_indices]
    return train_samples, val_samples


def _audit_and_raise(
    train_samples: Sequence[Sample5],
    val_samples: Sequence[Sample5],
    test_samples: Sequence[Sample5],
    class_names: Sequence[str],
    source_domains: Sequence[str],
    holdout_domain: str,
) -> DatasetAudit:
    if len(class_names) <= 1:
        raise ValueError(
            "benchmark build failed: only one class was parsed. This usually means the dataset root was wrong or labels were mis-read."
        )
    if not train_samples:
        raise ValueError("benchmark build failed: source train split is empty.")
    if not test_samples:
        raise ValueError("benchmark build failed: holdout test split is empty.")

    train_labels = {int(x[1]) for x in train_samples}
    test_labels = {int(x[1]) for x in test_samples}
    val_labels = {int(x[1]) for x in val_samples}

    if len(train_labels) <= 1:
        raise ValueError("benchmark build failed: train split collapsed to one class.")
    if len(test_labels) <= 1:
        raise ValueError("benchmark build failed: test split collapsed to one class.")

    max_label = len(class_names) - 1
    for name, label_set in {"train": train_labels, "val": val_labels, "test": test_labels}.items():
        if not label_set:
            continue
        if min(label_set) < 0 or max(label_set) > max_label:
            raise ValueError(f"benchmark build failed: {name} labels fall outside [0, {max_label}].")

    source_domain_set = set(source_domains)
    if holdout_domain in source_domain_set:
        raise ValueError("benchmark build failed: holdout domain leaked into source domain list.")

    train_domains = {x[2] for x in train_samples}
    val_domains = {x[2] for x in val_samples}
    test_domains = {x[2] for x in test_samples}
    if holdout_domain in train_domains or holdout_domain in val_domains:
        raise ValueError("benchmark build failed: holdout domain leaked into train/val splits.")
    if test_domains != {holdout_domain}:
        raise ValueError("benchmark build failed: test split contains non-holdout domains.")

    return DatasetAudit(
        class_names=list(class_names),
        source_domains=list(source_domains),
        holdout_domain=str(holdout_domain),
        n_train=len(train_samples),
        n_val=len(val_samples),
        n_test=len(test_samples),
    )


def _build_imagefolder_benchmark_strict(cfg: TAFDGConfig) -> FederatedBenchmark:
    domains, domain_samples, class_names = _scan_imagefolder_domains(cfg.data_root, cfg.domains)
    holdout_domain = cfg.holdout_domain or domains[cfg.holdout_index]
    if holdout_domain not in domains:
        raise ValueError(f"Holdout domain {holdout_domain} not found in {domains}")

    label_map = {name: idx for idx, name in enumerate(class_names)}
    source_domains = [d for d in domains if d != holdout_domain]

    train_transform = BasicImageTransform(cfg.image_size, train=True, allow_hflip=cfg.random_hflip)
    eval_transform = BasicImageTransform(cfg.image_size, train=False, allow_hflip=False)

    source_samples: List[Sample5] = []
    test_samples: List[Sample5] = []
    for domain in domains:
        packed = [
            (path, label_map[label_name], domain, domain, None)
            for path, label_name in domain_samples[domain]
        ]
        if domain == holdout_domain:
            test_samples.extend(packed)
        else:
            source_samples.extend(packed)

    train_samples, val_samples = _split_per_domain_per_class(source_samples, cfg.val_ratio, cfg.seed)
    _audit_and_raise(train_samples, val_samples, test_samples, class_names, source_domains, holdout_domain)

    train_dataset = StyledVisionDataset(train_samples, transform=train_transform, base_seed=cfg.seed)
    val_dataset = StyledVisionDataset(val_samples, transform=eval_transform, base_seed=cfg.seed + 123) if val_samples else None
    test_dataset = StyledVisionDataset(test_samples, transform=eval_transform, base_seed=cfg.seed + 997)
    client_datasets = _build_client_subsets_from_train_samples(train_dataset, train_samples, source_domains, cfg)

    return FederatedBenchmark(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        source_domains=source_domains,
        holdout_domain=holdout_domain,
        class_names=class_names,
    )


def _resolve_preset_samples_strict(cfg: TAFDGConfig, preset_name: str) -> Tuple[List[str], Dict[str, List[BasePresetSample]]]:
    if preset_name == "gtsrb":
        root = _ensure_gtsrb_available(cfg)
        train_root = _find_gtsrb_training_root(root)
        if train_root is None:
            raise FileNotFoundError("GTSRB training root was not found after download/extraction.")
        return _collect_classification_samples(train_root)

    if preset_name == "tt100k":
        root = _ensure_tt100k_available(cfg)
        try:
            return _load_tt100k_samples(root)
        except Exception:
            return _collect_classification_samples(root)

    if preset_name == "miotcd":
        root = _ensure_miotcd_available(cfg)
        try:
            return _load_csv_label_samples(root)
        except Exception:
            return _collect_classification_samples(root)

    raise ValueError(f"Unsupported preset dataset: {preset_name}")


def _build_generated_feddg_benchmark_strict(cfg: TAFDGConfig, preset_name: str) -> FederatedBenchmark:
    class_names, class_to_samples = _resolve_preset_samples_strict(cfg, preset_name)
    preset = PRESET_DATASETS[preset_name]
    all_domains = list(cfg.domains or preset["default_domains"])
    if len(all_domains) < 2:
        raise ValueError("At least two domains are required for federated domain generalization.")

    holdout_domain = cfg.holdout_domain or all_domains[cfg.holdout_index]
    if holdout_domain not in all_domains:
        raise ValueError(f"Holdout domain {holdout_domain} not found in {all_domains}")
    source_domains = [d for d in all_domains if d != holdout_domain]

    label_map = {name: idx for idx, name in enumerate(class_names)}
    train_transform = BasicImageTransform(cfg.image_size, train=True, allow_hflip=cfg.random_hflip)
    eval_transform = BasicImageTransform(cfg.image_size, train=False, allow_hflip=False)

    rng = np.random.default_rng(cfg.seed)
    source_samples: List[Sample5] = []
    test_samples: List[Sample5] = []

    for class_name in class_names:
        records = list(class_to_samples[class_name])
        rng.shuffle(records)
        buckets = [[] for _ in all_domains]
        for idx, record in enumerate(records):
            buckets[idx % len(all_domains)].append(record)

        for domain_name, bucket in zip(all_domains, buckets):
            for record in bucket:
                packed = (record.path, label_map[class_name], domain_name, domain_name, record.bbox)
                if domain_name == holdout_domain:
                    test_samples.append(packed)
                else:
                    source_samples.append(packed)

    train_samples, val_samples = _split_per_domain_per_class(source_samples, cfg.val_ratio, cfg.seed)
    _audit_and_raise(train_samples, val_samples, test_samples, class_names, source_domains, holdout_domain)

    train_dataset = StyledVisionDataset(train_samples, transform=train_transform, base_seed=cfg.seed)
    val_dataset = StyledVisionDataset(val_samples, transform=eval_transform, base_seed=cfg.seed + 123) if val_samples else None
    test_dataset = StyledVisionDataset(test_samples, transform=eval_transform, base_seed=cfg.seed + 997)
    client_datasets = _build_client_subsets_from_train_samples(train_dataset, train_samples, source_domains, cfg)

    return FederatedBenchmark(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        source_domains=source_domains,
        holdout_domain=holdout_domain,
        class_names=class_names,
    )


def build_benchmark(cfg: TAFDGConfig) -> FederatedBenchmark:
    if cfg.dataset == "synthetic":
        from .data import _build_synthetic_benchmark
        return _build_synthetic_benchmark(cfg)
    if cfg.dataset == "imagefolder":
        return _build_imagefolder_benchmark_strict(cfg)
    if cfg.dataset in PRESET_DATASETS:
        return _build_generated_feddg_benchmark_strict(cfg)
    raise ValueError(f"Unsupported dataset type: {cfg.dataset}")
