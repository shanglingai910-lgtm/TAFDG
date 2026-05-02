This build focuses on improving experiment success rate while preserving the TAFDG skeleton (local alignment + server alignment + Top-K + round-aware DP).

Key rewrites in this revision:

1. Client partition logic is now actually wired to preset_split_mode.
   - Previous code always used Dirichlet partitioning even when preset_split_mode='balanced'.
   - The new code really supports balanced / sequential partitioning, which avoids extreme per-client label collapse and makes early optimization much more stable.

2. The trainer now uses a staged warmup.
   - method_warmup_rounds runs plain FedAvg-style aggregation first.
   - After the warmup, local alignment, Top-K and DP are turned on again.
   - This keeps the TAFDG framework but avoids the "all components start too early and learning never gets off the ground" problem.

3. Local alignment no longer freezes the whole round.
   - Alignment reference is blended from the previous clean global delta and the current round mean clean client delta.
   - If too few clients pass tau, the trainer rescues the top aligned clients instead of producing an all-zero update round.

4. CPU execution is safer by default.
   - num_threads is capped for CPU-heavy quick tests so that smoke runs do not stall because of thread oversubscription.

Recommended order:
- quickstart
- sanity_* (verify parsing / labels)
- tafdg_*_stable
- tafdg_*_strong
