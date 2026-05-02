from __future__ import annotations

import copy
import json
import os

import torch

from tafdg.config import TAFDGConfig
from tafdg.data import PRESET_DATASETS, build_benchmark, default_domains_for_dataset, summarize_benchmark
from tafdg.trainer import TAFDGTrainer
from tafdg.utils import save_json, set_seed


def _run_single(cfg: TAFDGConfig) -> dict:
    set_seed(cfg.seed, cfg.num_threads)
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save_json(os.path.join(cfg.output_dir, 'config.json'))
    benchmark = build_benchmark(cfg)
    benchmark_summary = summarize_benchmark(benchmark)
    save_json(benchmark_summary, os.path.join(cfg.output_dir, 'benchmark_summary.json'))
    print(json.dumps({'benchmark_summary': benchmark_summary}, indent=2, ensure_ascii=False))

    trainer = TAFDGTrainer(cfg, benchmark)
    summary = trainer.train()
    print(
        json.dumps(
            {
                'dataset': cfg.dataset,
                'holdout_domain': summary['holdout_domain'],
                'best_val_accuracy': summary.get('best_val_accuracy', 0.0),
                'best_test_accuracy': summary['best_test_accuracy'],
                'final_test_accuracy': summary['final_test_accuracy'],
                'last10_mean_accuracy': summary['last10_mean_accuracy'],
                'avg_communication_mb': summary['avg_communication_mb'],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return summary


def _resolve_all_holdouts(cfg: TAFDGConfig) -> list[str]:
    if cfg.dataset == 'imagefolder':
        if cfg.domains is not None:
            return cfg.domains
        if not cfg.data_root:
            raise ValueError("When dataset='imagefolder', data_root is required if domains is not set.")
        return sorted(
            [
                d
                for d in os.listdir(cfg.data_root)
                if os.path.isdir(os.path.join(cfg.data_root, d))
            ]
        )
    if cfg.dataset == 'synthetic':
        return cfg.domains or default_domains_for_dataset(cfg.dataset, num_domains=cfg.num_domains)
    if cfg.dataset in PRESET_DATASETS:
        return cfg.domains or default_domains_for_dataset(cfg.dataset, num_domains=cfg.num_domains)
    raise ValueError(f'Unsupported dataset type: {cfg.dataset}')


def _auto_num_workers(profile: str) -> int:
    if os.name == 'nt':
        # Windows repeatedly spawning DataLoader workers for many clients is a major slowdown.
        return 0
    cpu_count = os.cpu_count() or 4
    if 'strong' in profile or profile.startswith('tafdg_'):
        return min(4, max(2, cpu_count // 4))
    return min(2, max(0, cpu_count // 8))


def build_user_config(profile: str = 'tafdg_gtsrb_stable') -> dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    workers = _auto_num_workers(profile)

    common = dict(
        seed=42,
        device=device,
        num_workers=workers,
        num_threads=1 if device == 'cpu' else 0,
        lr=5e-3,
        weight_decay=5e-4,
        momentum=0.9,
        tau=0.02,
        align_warmup_rounds=5,
        method_warmup_rounds=5,
        min_kept_ratio=0.25,
        min_kept_clients=2,
        align_rescue_scale=0.35,
        reference_blend=0.5,
        reference_reset_cosine=-0.10,
        topk_ratio=0.02,
        clip_norm=2.0,
        epsilon=80.0,
        delta=1e-5,
        dp_mode='practical',
        sigma_max=0.6,
        dp_noise_scale_mode='l2_normalized',
        disable_local_align=False,
        disable_server_align=False,
        disable_topk=False,
        disable_dp=False,
        random_hflip=True,
        preset_split_mode='balanced',
        val_ratio=0.1,
        dirichlet_alpha=1.0,
        local_epochs=2,
        save_last=True,
        checkpoint_every=10,
        auto_download=True,
        dataset_cache_dir='datasets',
        preset_use_official_test=False,
        min_class_total_samples=2,
    )

    profiles = {
        'quickstart': dict(
            dataset='synthetic',
            data_root='',
            domains=None,
            holdout_domain=None,
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/quickstart_synthetic',
            auto_download=False,
            num_clients=12,
            clients_per_round=1.0,
            rounds=12,
            batch_size=32,
            image_size=32,
            num_classes=10,
            num_domains=4,
            model='tinycnn',
            checkpoint_every=0,
            lr=1e-2,
            method_warmup_rounds=3,
            align_warmup_rounds=3,
            topk_ratio=0.1,
            synthetic_samples_per_class=24,
            synthetic_noise=0.08,
            synthetic_style_strength=0.35,
        ),
        'sanity_gtsrb': dict(
            dataset='gtsrb',
            data_root='',
            domains=['day', 'night', 'fog', 'motion'],
            holdout_domain='fog',
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/sanity_gtsrb',
            num_clients=20,
            clients_per_round=1.0,
            rounds=12,
            batch_size=64,
            image_size=64,
            num_classes=43,
            num_domains=4,
            model='resnet18',
            checkpoint_every=0,
            disable_local_align=True,
            disable_server_align=True,
            disable_topk=True,
            disable_dp=True,
        ),
        'tafdg_gtsrb_stable': dict(
            dataset='gtsrb',
            data_root='',
            domains=['day', 'night', 'fog', 'motion'],
            holdout_domain='fog',
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/tafdg_gtsrb_stable',
            num_clients=24,
            clients_per_round=0.5,
            rounds=30,
            batch_size=64,
            image_size=64,
            num_classes=43,
            num_domains=4,
            model='resnet18',
            checkpoint_every=5,
        ),
        'tafdg_gtsrb_strong': dict(
            dataset='gtsrb',
            data_root='',
            domains=['day', 'night', 'fog', 'motion'],
            holdout_domain='fog',
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/tafdg_gtsrb_strong',
            num_clients=36,
            clients_per_round=0.5,
            rounds=60,
            batch_size=64,
            image_size=64,
            num_classes=43,
            num_domains=4,
            model='resnet18',
            checkpoint_every=5,
        ),
        'sanity_tt100k': dict(
            dataset='tt100k',
            data_root='',
            domains=['day', 'night', 'rain', 'fog'],
            holdout_domain='fog',
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/sanity_tt100k',
            num_clients=20,
            clients_per_round=1.0,
            rounds=12,
            batch_size=64,
            image_size=64,
            num_classes=128,
            num_domains=4,
            model='resnet18',
            checkpoint_every=0,
            disable_local_align=True,
            disable_server_align=True,
            disable_topk=True,
            disable_dp=True,
            min_class_total_samples=4,
        ),
        'tafdg_tt100k_stable': dict(
            dataset='tt100k',
            data_root='',
            domains=['day', 'night', 'rain', 'fog'],
            holdout_domain='fog',
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/tafdg_tt100k_stable',
            num_clients=32,
            clients_per_round=0.5,
            rounds=40,
            batch_size=64,
            image_size=96,
            num_classes=128,
            num_domains=4,
            model='resnet18',
            checkpoint_every=5,
            min_class_total_samples=4,
        ),
        'tafdg_tt100k_strong': dict(
            dataset='tt100k',
            data_root='',
            domains=['day', 'night', 'rain', 'fog'],
            holdout_domain='fog',
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/tafdg_tt100k_strong',
            num_clients=80,
            clients_per_round=0.5,
            rounds=60,
            batch_size=64,
            image_size=96,
            num_classes=128,
            num_domains=4,
            model='resnet18',
            checkpoint_every=5,
            min_class_total_samples=4,
        ),
        'sanity_miotcd': dict(
            dataset='miotcd',
            data_root='',
            domains=['day', 'night', 'fog', 'jpeg'],
            holdout_domain='fog',
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/sanity_miotcd',
            num_clients=20,
            clients_per_round=1.0,
            rounds=12,
            batch_size=64,
            image_size=64,
            num_classes=11,
            num_domains=4,
            model='resnet18',
            checkpoint_every=0,
            disable_local_align=True,
            disable_server_align=True,
            disable_topk=True,
            disable_dp=True,
            min_class_total_samples=4,
        ),
        'tafdg_miotcd_stable': dict(
            dataset='miotcd',
            data_root='',
            domains=['day', 'night', 'fog', 'jpeg'],
            holdout_domain='fog',
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/tafdg_miotcd_stable',
            num_clients=24,
            clients_per_round=0.5,
            rounds=30,
            batch_size=64,
            image_size=96,
            num_classes=11,
            num_domains=4,
            model='resnet18',
            checkpoint_every=5,
            min_class_total_samples=4,
        ),
        'tafdg_miotcd_strong': dict(
            dataset='miotcd',
            data_root='',
            domains=['day', 'night', 'fog', 'jpeg'],
            holdout_domain='fog',
            holdout_index=0,
            run_all_holdouts=False,
            output_dir='outputs/tafdg_miotcd_strong',
            num_clients=60,
            clients_per_round=0.5,
            rounds=50,
            batch_size=64,
            image_size=96,
            num_classes=11,
            num_domains=4,
            model='resnet18',
            checkpoint_every=5,
            min_class_total_samples=4,
        ),
    }

    if profile not in profiles:
        available = ', '.join(sorted(profiles))
        raise ValueError(f'Unsupported profile: {profile}. Available profiles: {available}')

    if device == 'cpu':
        # CPU fallback: keep the TAFDG skeleton, but shrink per-round work to avoid appearing frozen.
        for name, patch in {
            'tafdg_gtsrb_stable': dict(clients_per_round=0.25, local_epochs=1, rounds=20, checkpoint_every=0),
            'tafdg_gtsrb_strong': dict(clients_per_round=0.33, local_epochs=1, rounds=30, checkpoint_every=0),
            'tafdg_tt100k_stable': dict(clients_per_round=0.25, local_epochs=1, rounds=24, checkpoint_every=0),
            'tafdg_tt100k_strong': dict(clients_per_round=0.33, local_epochs=1, rounds=36, checkpoint_every=0),
            'tafdg_miotcd_stable': dict(clients_per_round=0.25, local_epochs=1, rounds=24, checkpoint_every=0),
            'tafdg_miotcd_strong': dict(clients_per_round=0.33, local_epochs=1, rounds=36, checkpoint_every=0),
        }.items():
            if name in profiles:
                profiles[name].update(patch)

    config = dict(common)
    config.update(profiles[profile])
    return config


def main() -> None:
    """
    Focused profiles for the current project:
    - sanity_*: only verify parsing / labels / split integrity
    - tafdg_*_stable: Windows-friendly default
    - tafdg_*_strong: larger runs after stable profile works
    """

    PROFILE = 'tafdg_gtsrb_stable'
    USER_CONFIG = build_user_config(PROFILE)

    # Typical customizations:
    # USER_CONFIG.update(dict(dataset='gtsrb', holdout_domain='night'))
    # USER_CONFIG.update(dict(dataset='tt100k', data_root='/path/to/tt100k'))
    # USER_CONFIG.update(dict(dataset='miotcd', data_root='/path/to/miotcd'))
    # USER_CONFIG.update(dict(run_all_holdouts=False))

    run_all_holdouts = bool(USER_CONFIG.pop('run_all_holdouts', False))
    cfg = TAFDGConfig(**USER_CONFIG)

    if not run_all_holdouts:
        _run_single(cfg)
        return

    domains = _resolve_all_holdouts(cfg)
    all_results: dict[str, dict] = {}
    for holdout_domain in domains:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.holdout_domain = holdout_domain
        run_cfg.output_dir = os.path.join(cfg.output_dir, f'holdout_{holdout_domain}')
        all_results[holdout_domain] = _run_single(run_cfg)

    aggregate = {
        'dataset': cfg.dataset,
        'holdouts': {
            name: {
                'best_val_accuracy': result.get('best_val_accuracy', 0.0),
                'best_test_accuracy': result['best_test_accuracy'],
                'final_test_accuracy': result['final_test_accuracy'],
                'last10_mean_accuracy': result['last10_mean_accuracy'],
                'avg_communication_mb': result['avg_communication_mb'],
            }
            for name, result in all_results.items()
        },
    }
    save_json(aggregate, os.path.join(cfg.output_dir, 'all_holdouts_summary.json'))
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
