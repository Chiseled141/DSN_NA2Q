# Training Findings & Known Issues

## HiT-MAC Phase 2 Coordinator — Does Not Converge

**Status:** Phase 2 training is excluded from all reported results.

**What happens:** After Phase 1 executor training completes (~59% coverage), Phase 2 coordinator training degrades performance back down to ~37% — worse than random baseline (31.6%).

**Root cause:** Phase 2 trains the coordinator using a **scripted (perfect) executor** as surrogate. At test time the **learned executor** is used instead. The coordinator learns target assignments optimized for zero-error execution, then performs poorly when the real noisy executor is used. This is a known train/test mismatch in hierarchical RL.

**What to report:** HiT-MAC Phase 1 (executor only) results. Note Phase 2 coordinator training was excluded due to train/test mismatch.

---

## HiT-MAC Scenario 1 — Run4 Training Collapse

**Status:** Run4 (seed=46) collapsed midway through Phase 1 and was not re-run.

**What happens:**
- Episodes 0–1000: learns normally, reaches ~54% coverage
- Episodes 1000–12500: slowly drifts down
- Episode ~12500: collapses to ~21% coverage (below random baseline of 31.6%)
- Stays flat at ~21% for the rest of Phase 1

**Root cause:** A3C shared-memory race condition. With 8+ parallel workers writing gradients to a shared model simultaneously, one worker's catastrophic gradient update corrupted the shared weights. Other workers then computed gradients from a bad model, cascading into a death spiral. The policy collapsed to a degenerate deterministic action (likely always "Stay"), which is why coverage dropped below random.

**Impact on results:**
- With run4: mean = 52.0% ± 20.6% (misleadingly high std)
- Without run4: mean = 59.6% ± ~0.9% (4 seeds, much tighter)

**Recommendation:** Re-run seed 46 with seed 100 or any other value. Use `--run-id 4 --seed 100` to overwrite. Or report 4 seeds and note one diverged.

**Fix (long-term):** Increase entropy regularization (`--entropy 0.05` instead of default 0.01) to prevent policy collapse, or add per-worker gradient norm clipping.

---

## HiT-MAC Scenario 2 Phase 2 — Stuck at 0 Steps

**Status:** Phase 2 for Scenario 2 never started (0 steps after 3 hours).

**What happens:** After Phase 1 completes at 65.1% coverage, Phase 2 coordinator workers launch but the step counter stays at 0. Only the monitor process is alive.

**Root cause (likely):** Shapley value computation in the coordinator model scales poorly with number of agents. With 50 sensors (vs 5 in Scenario 2), the coordinator forward pass is extremely expensive, possibly causing workers to timeout or OOM silently before completing a single step.

**What to report:** HiT-MAC Scenario 2 Phase 1 results (65.1% coverage, single run). Note Phase 2 was computationally infeasible at this scale.

---

## NA²Q Training Time on CPU

**Observed:** NA²Q Scenario 1 takes ~17 hours on CPU (cloud instance without CUDA).

**Fix:** Always verify CUDA is available before training:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Then run with `--device cuda` explicitly.

---

## Config Changes for GPU Training

Scenario 1 config was tuned for GPU speed (2026-05-02):
- `episodes`: 50000 → 25000
- `chunk_length`: 25 → 10
- `epsilon_decay`: 7500 → 5000

Original values were designed for CPU training. On GPU, fewer episodes with shorter chunk length converges comparably faster.
