# TAFDG code rewrite notes

## Why the previous run looked stuck

From the uploaded `round_metrics.csv`, the previous pipeline showed two clear failure patterns:

1. `mean_local_cosine` turned negative after local alignment started, and `kept_batch_ratio` often dropped to `0.125` or even `0.0`.
   - This means most client updates were being filtered out after alignment warmup.
   - Several rounds therefore became near-zero-update rounds, so the loss barely moved.

2. The code path was configured as if `preset_split_mode='balanced'`, but the actual partition function still used Dirichlet partitioning.
   - This created much stronger per-client class collapse than intended.
   - The resulting proxy-domain clients were harder to optimize and made convergence much less stable.

## What was rewritten

### 1. Client partition logic is now consistent with the config

`preset_split_mode='balanced'` now really performs balanced per-domain, per-class round-robin partitioning.

### 2. A staged warmup was added

`method_warmup_rounds` lets the model learn a usable recognition direction before full TAFDG constraints are enabled.

### 3. Local alignment now has a rescue path

If too few clients pass `tau`, the trainer keeps the top aligned clients instead of allowing the round to collapse into all-zero updates.

### 4. The alignment reference is more robust

The reference direction is no longer taken from only the stale previous delta. It is blended with the current round mean client delta and resets to the current mean when the old reference is clearly contradictory.

### 5. CPU quick tests are safer

Default CPU `num_threads` was reduced to avoid oversubscription stalls in smoke runs.

## Suggested run order

1. `quickstart`
2. `sanity_gtsrb` / `sanity_tt100k` / `sanity_miotcd`
3. `tafdg_*_stable`
4. `tafdg_*_strong`
