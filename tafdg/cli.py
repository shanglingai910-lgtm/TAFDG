from __future__ import annotations

import argparse
import copy
import json
import os
from typing import List

from .config import TAFDGConfig
from .data import PRESET_DATASETS, available_dataset_names, build_benchmark, default_domains_for_dataset
from .trainer import TAFDGTrainer
from .utils import save_json, set_seed



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TAFDG: Two-end aligned federated domain generalization")
    parser.add_argument("--dataset", choices=available_dataset_names(), default="synthetic")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--holdout-domain", type=str, default=None)
    parser.add_argument("--holdout-index", type=int, default=0)
    parser.add_argument("--all-holdouts", action="store_true")

    parser.add_argument("--output-dir", type=str, default="outputs/tafdg")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-domains", type=int, default=4)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=float, default=1.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)

    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--model", choices=["resnet18", "tinycnn"], default="resnet18")

    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--align-warmup-rounds", type=int, default=1)
    parser.add_argument("--method-warmup-rounds", type=int, default=0)
    parser.add_argument("--min-kept-ratio", type=float, default=0.25)
    parser.add_argument("--min-kept-clients", type=int, default=1)
    parser.add_argument("--align-rescue-scale", type=float, default=0.35)
    parser.add_argument("--reference-blend", type=float, default=0.5)
    parser.add_argument("--reference-reset-cosine", type=float, default=-0.05)
    parser.add_argument("--topk-ratio", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--dp-mode", choices=["practical", "exact"], default="practical")
    parser.add_argument("--sigma-max", type=float, default=5.0)
    parser.add_argument("--dp-noise-scale-mode", choices=["per_coordinate", "l2_normalized"], default="l2_normalized")

    parser.add_argument("--disable-local-align", action="store_true")
    parser.add_argument("--disable-server-align", action="store_true")
    parser.add_argument("--disable-topk", action="store_true")
    parser.add_argument("--disable-dp", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--random-hflip", action="store_true")
    parser.add_argument("--preset-split-mode", choices=["balanced", "sequential"], default="balanced")
    parser.add_argument("--auto-download", dest="auto_download", action="store_true")
    parser.add_argument("--no-auto-download", dest="auto_download", action="store_false")
    parser.set_defaults(auto_download=True)
    parser.add_argument("--dataset-cache-dir", type=str, default="datasets")

    parser.add_argument("--synthetic-samples-per-class", type=int, default=64)
    parser.add_argument("--synthetic-noise", type=float, default=0.08)
    parser.add_argument("--synthetic-style-strength", type=float, default=0.35)
    return parser



def args_to_config(args: argparse.Namespace) -> TAFDGConfig:
    cfg = TAFDGConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        domains=args.domains,
        holdout_domain=args.holdout_domain,
        holdout_index=args.holdout_index,
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_classes=args.num_classes,
        num_domains=args.num_domains,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        dirichlet_alpha=args.dirichlet_alpha,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        model=args.model,
        tau=args.tau,
        align_warmup_rounds=args.align_warmup_rounds,
        method_warmup_rounds=args.method_warmup_rounds,
        min_kept_ratio=args.min_kept_ratio,
        min_kept_clients=args.min_kept_clients,
        align_rescue_scale=args.align_rescue_scale,
        reference_blend=args.reference_blend,
        reference_reset_cosine=args.reference_reset_cosine,
        topk_ratio=args.topk_ratio,
        clip_norm=args.clip_norm,
        epsilon=args.epsilon,
        delta=args.delta,
        dp_mode=args.dp_mode,
        sigma_max=args.sigma_max,
        dp_noise_scale_mode=args.dp_noise_scale_mode,
        disable_local_align=args.disable_local_align,
        disable_server_align=args.disable_server_align,
        disable_topk=args.disable_topk,
        disable_dp=args.disable_dp,
        seed=args.seed,
        device=args.device or TAFDGConfig.device,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
        random_hflip=args.random_hflip,
        auto_download=args.auto_download,
        dataset_cache_dir=args.dataset_cache_dir,
        synthetic_samples_per_class=args.synthetic_samples_per_class,
        synthetic_noise=args.synthetic_noise,
        synthetic_style_strength=args.synthetic_style_strength,
        preset_split_mode=args.preset_split_mode,
        checkpoint_every=args.checkpoint_every,
    )
    return cfg



def _run_single(cfg: TAFDGConfig) -> dict:
    set_seed(cfg.seed, cfg.num_threads)
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save_json(os.path.join(cfg.output_dir, "config.json"))
    benchmark = build_benchmark(cfg)
    trainer = TAFDGTrainer(cfg, benchmark)
    summary = trainer.train()
    print(
        json.dumps(
            {
                "dataset": cfg.dataset,
                "holdout_domain": summary["holdout_domain"],
                "best_test_accuracy": summary["best_test_accuracy"],
                "final_test_accuracy": summary["final_test_accuracy"],
                "last10_mean_accuracy": summary["last10_mean_accuracy"],
                "avg_communication_mb": summary["avg_communication_mb"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return summary



def _resolve_all_holdouts(cfg: TAFDGConfig) -> List[str]:
    if cfg.dataset == "imagefolder":
        if cfg.domains is not None:
            return cfg.domains
        if not cfg.data_root:
            raise ValueError("--data-root is required for --dataset imagefolder")
        return sorted([d for d in os.listdir(cfg.data_root) if os.path.isdir(os.path.join(cfg.data_root, d))])
    if cfg.dataset == "synthetic":
        return cfg.domains or default_domains_for_dataset(cfg.dataset, num_domains=cfg.num_domains)
    if cfg.dataset in PRESET_DATASETS:
        return cfg.domains or default_domains_for_dataset(cfg.dataset, num_domains=cfg.num_domains)
    raise ValueError(f"Unsupported dataset type: {cfg.dataset}")



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)

    if not args.all_holdouts:
        _run_single(cfg)
        return

    domains = _resolve_all_holdouts(cfg)
    all_results = {}
    for holdout_domain in domains:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.holdout_domain = holdout_domain
        run_cfg.output_dir = os.path.join(cfg.output_dir, f"holdout_{holdout_domain}")
        all_results[holdout_domain] = _run_single(run_cfg)

    aggregate = {
        "dataset": cfg.dataset,
        "holdouts": {
            name: {
                "best_test_accuracy": result["best_test_accuracy"],
                "final_test_accuracy": result["final_test_accuracy"],
                "last10_mean_accuracy": result["last10_mean_accuracy"],
                "avg_communication_mb": result["avg_communication_mb"],
            }
            for name, result in all_results.items()
        },
    }
    save_json(aggregate, os.path.join(cfg.output_dir, "all_holdouts_summary.json"))
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
