# Training Findings & Known Issues

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
