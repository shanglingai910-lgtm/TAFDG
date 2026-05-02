from __future__ import annotations

import os
import torch


def _auto_num_workers(profile: str) -> int:
    cpu_count = os.cpu_count() or 4
    if profile.startswith("paper"):
        return min(8, max(4, cpu_count // 4))
    return min(4, max(2, cpu_count // 8))


def build_user_config(profile: str = "quickstart") -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = _auto_num_workers(profile)

    common = dict(
        seed=42,
        device=device,
        num_workers=workers,
        num_threads=0,
        lr=1e-3,
        weight_decay=5e-4,
        momentum=0.9,
        tau=0.1,
        align_warmup_rounds=5,
        topk_ratio=0.1,
        clip_norm=5.0,
        epsilon=8.0,
        delta=1e-5,
        dp_mode="exact",
        sigma_max=1.5,
        disable_local_align=False,
        disable_server_align=False,
        disable_topk=False,
        disable_dp=False,
        random_hflip=True,
        preset_split_mode="balanced",
        val_ratio=0.1,
        dirichlet_alpha=1.0,
        local_epochs=1,
        save_last=True,
    )

    profiles = {
        "quickstart": dict(
            dataset="synthetic",
            data_root="",
            domains=None,
            holdout_domain=None,
            holdout_index=0,
            run_all_holdouts=False,
            output_dir="outputs/quickstart_synthetic",
            auto_download=False,
            dataset_cache_dir="datasets",
            num_clients=12,
            clients_per_round=0.5,
            rounds=8,
            batch_size=32,
            image_size=32,
            num_classes=10,
            num_domains=4,
            model="tinycnn",
            checkpoint_every=0,
            synthetic_samples_per_class=24,
            synthetic_noise=0.08,
            synthetic_style_strength=0.35,
        ),
        # 只做数据/标签体检，不用于论文结果。
        "sanity_gtsrb": dict(
            dataset="gtsrb",
            data_root="",
            domains=["day", "night", "fog", "motion"],
            holdout_domain="fog",
            holdout_index=0,
            run_all_holdouts=False,
            output_dir="outputs/sanity_gtsrb",
            auto_download=True,
            dataset_cache_dir="datasets",
            preset_use_official_test=False,
            num_clients=20,
            clients_per_round=0.5,
            rounds=10,
            batch_size=64,
            image_size=64,
            num_classes=43,
            num_domains=4,
            model="resnet18",
            checkpoint_every=0,
            disable_local_align=True,
            disable_server_align=True,
            disable_topk=True,
            disable_dp=True,
        ),
        # 保留 TAFDG 三件套，但规模略收缩，适合先跑通 GTSRB 代理域实验。
        "paper_compatible_gtsrb": dict(
            dataset="gtsrb",
            data_root="",
            domains=["day", "night", "fog", "motion"],
            holdout_domain="fog",
            holdout_index=0,
            run_all_holdouts=False,
            output_dir="outputs/paper_compatible_gtsrb",
            auto_download=True,
            dataset_cache_dir="datasets",
            preset_use_official_test=False,
            num_clients=50,
            clients_per_round=0.5,
            rounds=80,
            batch_size=128,
            image_size=64,
            num_classes=43,
            num_domains=4,
            model="resnet18",
            checkpoint_every=10,
        ),
        # 更接近论文：ResNet-18 / 100 轮 / 100 clients / tau=0.1 / Top-K=0.1d / round-aware DP。
        "paper_imagefolder": dict(
            dataset="imagefolder",
            data_root="/path/to/TerraInc_or_DomainNet",
            domains=["DomainA", "DomainB", "DomainC", "DomainD"],
            holdout_domain="DomainD",
            holdout_index=0,
            run_all_holdouts=False,
            output_dir="outputs/paper_imagefolder",
            auto_download=False,
            dataset_cache_dir="datasets",
            num_clients=100,
            clients_per_round=1.0,
            rounds=100,
            batch_size=128,
            image_size=64,
            num_classes=10,
            num_domains=4,
            model="resnet18",
            checkpoint_every=10,
            epsilon=5.0,
            clip_norm=1.0,
            sigma_max=5.0,
        ),
    }

    if profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unsupported profile: {profile}. Available profiles: {available}")

    config = dict(common)
    config.update(profiles[profile])
    return config
