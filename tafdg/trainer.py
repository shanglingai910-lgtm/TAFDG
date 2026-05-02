from __future__ import annotations

import copy
import csv
import math
import os
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TAFDGConfig
from .data import FederatedBenchmark
from .models import build_model
from .utils import (
    ParameterPacker,
    RoundMetrics,
    SparseUpdate,
    compute_round_epsilon,
    compute_sigma_from_budget,
    cosine_similarity,
    estimate_communication_bytes,
    save_json,
    topk_indices,
)


class TAFDGTrainer:
    def __init__(self, cfg: TAFDGConfig, benchmark: FederatedBenchmark) -> None:
        self.cfg = cfg
        self.benchmark = benchmark
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == 'cpu' else 'cpu')
        self.model = build_model(cfg.model, num_classes=len(benchmark.class_names), image_size=cfg.image_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.packer = ParameterPacker.from_model(self.model)
        eval_batch_size = max(1, min(max(cfg.batch_size * 2, cfg.batch_size), 512))
        self.test_loader = DataLoader(
            benchmark.test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        self.val_loader = None
        if benchmark.val_dataset is not None:
            self.val_loader = DataLoader(
                benchmark.val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
            )
        self.global_state = copy.deepcopy(self.model.state_dict())
        self.global_vector = self.packer.flatten_state_dict(self.global_state, device=self.device)
        self.previous_global_delta = torch.zeros_like(self.global_vector)
        self.previous_clean_global_delta = torch.zeros_like(self.global_vector)
        self.local_scratch_model = build_model(
            cfg.model,
            num_classes=len(benchmark.class_names),
            image_size=cfg.image_size,
        ).to(self.device)
        self.history: List[RoundMetrics] = []
        self.best_val_accuracy = float('-inf')
        self.best_test_accuracy = float('-inf')

    def _in_method_warmup(self, round_idx: int) -> bool:
        return round_idx < max(int(self.cfg.method_warmup_rounds), 0)

    def _local_align_active(self, round_idx: int) -> bool:
        return (
            not self.cfg.disable_local_align
            and not self._in_method_warmup(round_idx)
            and round_idx >= max(int(self.cfg.align_warmup_rounds), 0)
        )

    def _server_align_active(self, round_idx: int) -> bool:
        return (not self.cfg.disable_server_align) and (not self._in_method_warmup(round_idx))

    def _topk_active(self, round_idx: int) -> bool:
        # Keep Top-K active even during engineering warmup.
        # This is closer to the paper and avoids extremely heavy dense-update objects.
        return not self.cfg.disable_topk

    def _dp_active(self, round_idx: int) -> bool:
        return (not self.cfg.disable_dp) and (not self._in_method_warmup(round_idx))

    def _sync_model_from_vector(self) -> None:
        self.global_state = self.packer.vector_to_state_dict(self.global_vector, self.global_state, device=self.device)
        self.model.load_state_dict(self.global_state)

    def _resolve_alignment_reference(self, mean_raw_delta: torch.Tensor) -> torch.Tensor:
        prev_delta = self.previous_clean_global_delta
        prev_norm = float(torch.linalg.norm(prev_delta).item())
        current_norm = float(torch.linalg.norm(mean_raw_delta).item())
        if prev_norm < 1e-12:
            return mean_raw_delta.detach().clone()
        if current_norm < 1e-12:
            return prev_delta.detach().clone()

        prev_current_cos = cosine_similarity(prev_delta, mean_raw_delta)
        if prev_current_cos < float(self.cfg.reference_reset_cosine):
            return mean_raw_delta.detach().clone()

        blend = float(np.clip(self.cfg.reference_blend, 0.0, 1.0))
        ref_delta = blend * prev_delta + (1.0 - blend) * mean_raw_delta
        if float(torch.linalg.norm(ref_delta).item()) < 1e-12:
            return mean_raw_delta.detach().clone()
        return ref_delta.detach().clone()

    def _update_norm(self, update: SparseUpdate) -> float:
        if update.values.numel() == 0:
            return 0.0
        return float(torch.linalg.norm(update.values).item())

    def _accumulate_weighted_update(self, target: torch.Tensor, update: SparseUpdate, weight: float) -> torch.Tensor:
        if abs(weight) <= 1e-12 or update.values.numel() == 0:
            return target
        if update.sparse:
            indices = update.indices.long().to(target.device)
            values = update.values.to(target.device)
            target.index_add_(0, indices, values * float(weight))
        else:
            target.add_(update.values.to(target.device), alpha=float(weight))
        return target

    def _cosine_update_to_dense(self, update: SparseUpdate, dense: torch.Tensor) -> float:
        dense_norm = torch.linalg.norm(dense)
        if dense_norm.item() < 1e-12 or update.values.numel() == 0:
            return 0.0
        update_norm = torch.linalg.norm(update.values)
        if update_norm.item() < 1e-12:
            return 0.0
        if update.sparse:
            dot = torch.dot(update.values.to(dense.device), dense.index_select(0, update.indices.long().to(dense.device)))
        else:
            dot = torch.dot(update.values.to(dense.device), dense)
        return float(dot / (update_norm * dense_norm + 1e-12))

    def _compress(self, delta: torch.Tensor, round_idx: int) -> SparseUpdate:
        if torch.linalg.norm(delta).item() < 1e-12:
            return SparseUpdate(
                indices=torch.empty(0, dtype=torch.long, device=self.device),
                values=torch.empty(0, dtype=delta.dtype, device=self.device),
                total_size=delta.numel(),
                epsilon_t=0.0,
                sigma_t=0.0,
                communication_bytes=0,
                sparse=True,
            )

        sparse = self._topk_active(round_idx)
        if sparse:
            indices = topk_indices(delta, self.cfg.topk_ratio)
            values = delta.index_select(0, indices).clone()
        else:
            indices = torch.empty(0, dtype=torch.long, device=self.device)
            values = delta.clone()

        norm = torch.linalg.norm(values)
        if norm.item() > self.cfg.clip_norm:
            values.mul_(self.cfg.clip_norm / (norm + 1e-12))

        communication_bytes = estimate_communication_bytes(num_values=int(values.numel()), sparse=sparse)
        return SparseUpdate(
            indices=indices.detach(),
            values=values.detach(),
            total_size=delta.numel(),
            epsilon_t=0.0,
            sigma_t=0.0,
            communication_bytes=communication_bytes,
            sparse=sparse,
        )

    def _perturb_sparse_update(self, clean_update: SparseUpdate, round_idx: int) -> SparseUpdate:
        values = clean_update.values.clone()
        epsilon_t = 0.0
        sigma_t = 0.0
        if self._dp_active(round_idx) and values.numel() > 0:
            epsilon_t = compute_round_epsilon(
                epsilon=self.cfg.epsilon,
                round_idx=round_idx,
                total_rounds=self.cfg.rounds,
                mode=self.cfg.dp_mode,
            )
            sigma_t = compute_sigma_from_budget(
                epsilon_t=epsilon_t,
                delta=self.cfg.delta,
                sigma_max=self.cfg.sigma_max,
            )
            if self.cfg.dp_noise_scale_mode == 'l2_normalized':
                noise_std = float(sigma_t * self.cfg.clip_norm / math.sqrt(max(int(values.numel()), 1)))
            else:
                noise_std = float(sigma_t * self.cfg.clip_norm)
            values = values + torch.randn_like(values) * noise_std

        return SparseUpdate(
            indices=clean_update.indices.detach(),
            values=values.detach(),
            total_size=clean_update.total_size,
            epsilon_t=epsilon_t,
            sigma_t=sigma_t,
            communication_bytes=clean_update.communication_bytes,
            sparse=clean_update.sparse,
        )

    def _train_one_client(self, dataset) -> Dict[str, object]:
        local_model = self.local_scratch_model
        local_num_workers = 0 if os.name == 'nt' else self.cfg.num_workers
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.cfg.batch_size),
            shuffle=True,
            num_workers=local_num_workers,
            drop_last=False,
        )

        local_model.load_state_dict(copy.deepcopy(self.global_state))
        local_model.to(self.device)
        local_model.train()
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

        running_loss = 0.0
        num_batches = 0
        for _ in range(self.cfg.local_epochs):
            for batch in loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = local_model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                num_batches += 1

        local_vector = self.packer.flatten_model(local_model, device=self.device)
        raw_delta = local_vector - self.global_vector
        return {
            'raw_delta': raw_delta.detach(),
            'loss': running_loss / max(num_batches, 1),
        }

    def _evaluate_loader(self, loader: DataLoader | None) -> tuple[float, float]:
        if loader is None:
            return 0.0, 0.0
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.inference_mode():
            for x, y, _domain in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                total_loss += float(self.criterion(logits, y).item()) * y.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += int(y.size(0))
        return 100.0 * correct / max(total, 1), total_loss / max(total, 1)

    def _evaluate(self) -> tuple[float, float]:
        return self._evaluate_loader(self.test_loader)

    def _select_clients(self, round_idx: int) -> List[int]:
        n_clients = len(self.benchmark.client_datasets)
        if self.cfg.clients_per_round <= 1.0:
            m = max(1, int(round(n_clients * self.cfg.clients_per_round)))
        else:
            m = min(n_clients, int(round(self.cfg.clients_per_round)))
        rng = np.random.default_rng(self.cfg.seed + round_idx)
        return list(map(int, rng.choice(n_clients, size=m, replace=False)))

    def _save_checkpoint(self, round_idx: int, tag: str = '') -> None:
        checkpoint_dir = os.path.join(self.cfg.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        suffix = f'_{tag}' if tag else ''
        ckpt_path = os.path.join(checkpoint_dir, f'round_{round_idx + 1:04d}{suffix}.pt')
        torch.save(
            {
                'round_idx': round_idx,
                'state_dict': self.model.state_dict(),
                'history': [asdict(item) for item in self.history],
                'config': self.cfg.to_dict(),
                'holdout_domain': self.benchmark.holdout_domain,
                'source_domains': self.benchmark.source_domains,
                'class_names': self.benchmark.class_names,
            },
            ckpt_path,
        )

    def _write_history(self) -> None:
        rows = [asdict(item) for item in self.history]
        if not rows:
            return
        path = os.path.join(self.cfg.output_dir, 'round_metrics.csv')
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _build_client_result(self, raw_delta: torch.Tensor, loss: float, round_idx: int, local_cosine: float, keep_flag: float) -> Dict[str, object]:
        clean_update = self._compress(raw_delta, round_idx)
        noisy_update = self._perturb_sparse_update(clean_update, round_idx)
        return {
            'clean_update': clean_update,
            'noisy_update': noisy_update,
            'loss': float(loss),
            'mean_local_cosine': float(local_cosine),
            'kept_batch_ratio': float(keep_flag),
            'clean_update_norm': self._update_norm(clean_update),
            'noisy_update_norm': self._update_norm(noisy_update),
        }

    def train(self) -> dict:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        progress = tqdm(range(self.cfg.rounds), desc='TAFDG', leave=True)
        for round_idx in progress:
            round_start = time.time()
            selected_clients = self._select_clients(round_idx)
            align_active = self._local_align_active(round_idx)
            client_results: List[Dict[str, object]] = []
            local_cosines: List[float] = []
            keep_flags: List[float] = []

            if align_active:
                raw_client_results: List[Dict[str, object]] = []
                for client_pos, client_id in enumerate(selected_clients, start=1):
                    raw_result = self._train_one_client(self.benchmark.client_datasets[client_id])
                    raw_client_results.append(raw_result)
                    if client_pos == len(selected_clients) or client_pos % max(1, len(selected_clients) // 4) == 0:
                        progress.set_postfix(stage=f'local {client_pos}/{len(selected_clients)}')

                raw_deltas: List[torch.Tensor] = [item['raw_delta'] for item in raw_client_results if isinstance(item['raw_delta'], torch.Tensor)]
                if raw_deltas:
                    mean_raw_delta = torch.zeros_like(self.global_vector)
                    for raw_delta in raw_deltas:
                        mean_raw_delta = mean_raw_delta + raw_delta
                    mean_raw_delta = mean_raw_delta / max(len(raw_deltas), 1)
                else:
                    mean_raw_delta = torch.zeros_like(self.global_vector)

                ref_delta = self._resolve_alignment_reference(mean_raw_delta)
                ref_norm = float(torch.linalg.norm(ref_delta).item())
                aligned_deltas: List[torch.Tensor] = []

                if raw_deltas and ref_norm > 1e-12:
                    local_cosines = [float(cosine_similarity(raw_delta, ref_delta)) for raw_delta in raw_deltas]
                    keep_mask = [cos_val >= float(self.cfg.tau) for cos_val in local_cosines]
                    min_keep = max(int(self.cfg.min_kept_clients), int(math.ceil(len(raw_deltas) * float(self.cfg.min_kept_ratio))))
                    min_keep = min(len(raw_deltas), max(min_keep, 1))
                    if sum(keep_mask) < min_keep:
                        ranked_indices = list(np.argsort(np.asarray(local_cosines, dtype=np.float32))[::-1])
                        keep_mask = [False for _ in raw_deltas]
                        for idx in ranked_indices[:min_keep]:
                            keep_mask[int(idx)] = True

                    for raw_delta, cos_val, keep in zip(raw_deltas, local_cosines, keep_mask):
                        if not keep:
                            aligned_deltas.append(torch.zeros_like(raw_delta))
                            keep_flags.append(0.0)
                            continue
                        aligned_delta = raw_delta
                        if cos_val < float(self.cfg.tau):
                            rescue_scale = max(float(self.cfg.align_rescue_scale), 0.5 * (cos_val + 1.0))
                            aligned_delta = raw_delta * float(rescue_scale)
                        aligned_deltas.append(aligned_delta)
                        keep_flags.append(1.0)
                else:
                    aligned_deltas = [raw_delta.detach().clone() for raw_delta in raw_deltas]
                    if raw_deltas:
                        local_cosines = [1.0 for _ in raw_deltas]
                        keep_flags = [1.0 for _ in raw_deltas]

                for raw_result, aligned_delta, local_cosine, keep_flag in zip(raw_client_results, aligned_deltas, local_cosines, keep_flags):
                    client_results.append(
                        self._build_client_result(
                            raw_delta=aligned_delta,
                            loss=float(raw_result['loss']),
                            round_idx=round_idx,
                            local_cosine=float(local_cosine),
                            keep_flag=float(keep_flag),
                        )
                    )
            else:
                # Warmup: no local alignment. Compress immediately to avoid caching many dense raw deltas.
                for client_pos, client_id in enumerate(selected_clients, start=1):
                    raw_result = self._train_one_client(self.benchmark.client_datasets[client_id])
                    raw_delta = raw_result['raw_delta']
                    assert isinstance(raw_delta, torch.Tensor)
                    local_cosines.append(1.0)
                    keep_flags.append(1.0)
                    client_results.append(
                        self._build_client_result(
                            raw_delta=raw_delta,
                            loss=float(raw_result['loss']),
                            round_idx=round_idx,
                            local_cosine=1.0,
                            keep_flag=1.0,
                        )
                    )
                    # Free the full dense delta as early as possible in warmup rounds.
                    del raw_result
                    if client_pos == len(selected_clients) or client_pos % max(1, len(selected_clients) // 4) == 0:
                        progress.set_postfix(stage=f'local {client_pos}/{len(selected_clients)}')

            clean_updates: List[SparseUpdate] = []
            noisy_updates: List[SparseUpdate] = []
            for result in client_results:
                clean_update = result['clean_update']
                noisy_update = result['noisy_update']
                assert isinstance(clean_update, SparseUpdate)
                assert isinstance(noisy_update, SparseUpdate)
                clean_updates.append(clean_update)
                noisy_updates.append(noisy_update)

            if clean_updates:
                mean_clean_delta = torch.zeros_like(self.global_vector)
                inv_n = 1.0 / max(len(clean_updates), 1)
                for clean_update in clean_updates:
                    self._accumulate_weighted_update(mean_clean_delta, clean_update, inv_n)

                if (not self._server_align_active(round_idx)) or len(clean_updates) == 1 or torch.linalg.norm(mean_clean_delta).item() < 1e-12:
                    weights = np.ones(len(clean_updates), dtype=np.float32) / max(len(clean_updates), 1)
                    server_cosines = [0.0 if len(clean_updates) > 1 else 1.0 for _ in clean_updates]
                else:
                    sims_list = [self._cosine_update_to_dense(clean_update, mean_clean_delta) for clean_update in clean_updates]
                    sims = np.asarray(sims_list, dtype=np.float32)
                    server_cosines = sims.tolist()
                    weights = np.clip((sims + 1.0) / 2.0, 0.0, None)
                    if float(weights.sum()) <= 1e-12:
                        weights = np.ones_like(weights) / len(weights)
                    else:
                        weights = weights / weights.sum()

                clean_aggregated_delta = torch.zeros_like(self.global_vector)
                aggregated_delta = torch.zeros_like(self.global_vector)
                for weight, clean_update, noisy_update in zip(weights.tolist(), clean_updates, noisy_updates):
                    self._accumulate_weighted_update(clean_aggregated_delta, clean_update, float(weight))
                    self._accumulate_weighted_update(aggregated_delta, noisy_update, float(weight))
            else:
                server_cosines = []
                clean_aggregated_delta = torch.zeros_like(self.global_vector)
                aggregated_delta = torch.zeros_like(self.global_vector)

            self.global_vector = self.global_vector + aggregated_delta
            self.previous_global_delta = aggregated_delta.detach().clone()
            self.previous_clean_global_delta = clean_aggregated_delta.detach().clone()
            self._sync_model_from_vector()

            progress.set_postfix(stage='eval')
            val_acc, val_loss = self._evaluate_loader(self.val_loader)
            test_acc, test_loss = self._evaluate()

            train_losses = [float(item['loss']) for item in client_results]
            epsilons = [float(item['noisy_update'].epsilon_t) for item in client_results]
            sigmas = [float(item['noisy_update'].sigma_t) for item in client_results]
            comm_mb = sum(float(item['noisy_update'].communication_bytes) for item in client_results) / (1024.0 ** 2)
            clean_norms = [float(item['clean_update_norm']) for item in client_results]
            noisy_norms = [float(item['noisy_update_norm']) for item in client_results]
            last10_mean = float(np.mean([m.test_accuracy for m in self.history[-9:]] + [test_acc]))
            round_metrics = RoundMetrics(
                round_idx=round_idx + 1,
                train_loss=float(np.mean(train_losses)) if train_losses else 0.0,
                val_loss=float(val_loss),
                val_accuracy=float(val_acc),
                test_loss=float(test_loss),
                test_accuracy=float(test_acc),
                last10_mean_accuracy=last10_mean,
                mean_local_cosine=float(np.mean(local_cosines)) if local_cosines else 0.0,
                kept_batch_ratio=float(np.mean(keep_flags)) if keep_flags else 0.0,
                mean_server_cosine=float(np.mean(server_cosines)) if server_cosines else 0.0,
                mean_epsilon_t=float(np.mean(epsilons)) if epsilons else 0.0,
                mean_sigma_t=float(np.mean(sigmas)) if sigmas else 0.0,
                communication_mb=float(comm_mb),
                mean_clean_update_norm=float(np.mean(clean_norms)) if clean_norms else 0.0,
                mean_noisy_update_norm=float(np.mean(noisy_norms)) if noisy_norms else 0.0,
            )
            self.history.append(round_metrics)

            if round_metrics.kept_batch_ratio <= 0.25 and self._local_align_active(round_idx):
                print('[TAFDG][warn] local alignment kept very few client updates. The fallback rescue path was used to avoid a zero-update round.')
            if (
                round_idx >= 2
                and round_metrics.mean_noisy_update_norm > max(10.0 * round_metrics.mean_clean_update_norm, 1e-6)
            ):
                print(
                    '[TAFDG][warn] noisy update norm is dominating clean update norm. '
                    'Consider increasing epsilon, reducing sigma_max, lowering topk_ratio, or switching off DP for ablation.'
                )

            if self.val_loader is not None and val_acc >= self.best_val_accuracy:
                self.best_val_accuracy = float(val_acc)
                self._save_checkpoint(round_idx, tag='best_val')
            if test_acc >= self.best_test_accuracy:
                self.best_test_accuracy = float(test_acc)
                self._save_checkpoint(round_idx, tag='best_test')

            progress.set_postfix(
                val=f'{val_acc:.2f}' if self.val_loader is not None else '-',
                test=f'{test_acc:.2f}',
                local=f'{round_metrics.mean_local_cosine:.3f}',
                keep=f'{round_metrics.kept_batch_ratio:.2f}',
                nstd=f'{round_metrics.mean_noisy_update_norm:.3f}',
                sec=f'{time.time() - round_start:.1f}',
            )

            if self.cfg.checkpoint_every > 0 and ((round_idx + 1) % self.cfg.checkpoint_every == 0):
                self._save_checkpoint(round_idx)

        self._write_history()
        if self.cfg.save_last:
            self._save_checkpoint(self.cfg.rounds - 1, tag='last')

        best_acc = max((m.test_accuracy for m in self.history), default=0.0)
        summary = {
            'holdout_domain': self.benchmark.holdout_domain,
            'source_domains': self.benchmark.source_domains,
            'best_val_accuracy': max((m.val_accuracy for m in self.history), default=0.0),
            'best_test_accuracy': best_acc,
            'final_test_accuracy': self.history[-1].test_accuracy if self.history else 0.0,
            'last10_mean_accuracy': float(np.mean([m.test_accuracy for m in self.history[-10:]])) if self.history else 0.0,
            'avg_communication_mb': float(np.mean([m.communication_mb for m in self.history])) if self.history else 0.0,
            'avg_kept_batch_ratio': float(np.mean([m.kept_batch_ratio for m in self.history])) if self.history else 0.0,
            'avg_local_cosine': float(np.mean([m.mean_local_cosine for m in self.history])) if self.history else 0.0,
            'avg_clean_update_norm': float(np.mean([m.mean_clean_update_norm for m in self.history])) if self.history else 0.0,
            'avg_noisy_update_norm': float(np.mean([m.mean_noisy_update_norm for m in self.history])) if self.history else 0.0,
            'rounds': [asdict(item) for item in self.history],
        }
        save_json(summary, os.path.join(self.cfg.output_dir, 'summary.json'))
        return summary
