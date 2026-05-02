# Fast-path engineering notes

This revision keeps the TAFDG skeleton intact while changing the execution strategy so the code no longer *looks* frozen on practical hardware.

Main changes:

1. Top-K stays active during warmup.
   - Local/server alignment and DP can still be warmed up.
   - But update transport is compressed from round 1, which is closer to the paper and much lighter than dense warmup payloads.

2. Dense warmup no longer allocates full index tensors.
   - The previous implementation created `torch.arange(d)` for non-sparse updates.
   - With ResNet-18 this is a very large extra tensor and wastes both memory and time.

3. Sparse aggregation is streamed.
   - Server aggregation now uses `index_add_` instead of repeatedly materializing full dense client vectors.

4. Warmup rounds compress immediately.
   - When local alignment is off, client deltas are compressed right after local training instead of keeping many dense raw deltas in memory.

5. Progress heartbeat was added.
   - The outer progress bar now shows `stage=local i/n` and `stage=eval`, so long rounds no longer appear dead.

6. CPU profiles are auto-shrunk.
   - On CPU, stable/strong profiles automatically reduce `clients_per_round`, `local_epochs`, and checkpoint frequency.

Recommended interpretation:
- If the log line stays at `1/30` but the `stage=` field keeps changing, the process is alive.
- If `stage=eval` takes too long, evaluation is the bottleneck rather than training.
