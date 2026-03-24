# HiT-MAC: Hierarchical Twin-Actor Multi-Agent Coordination
## Slide Deck Outline

---

## Slide 1 — Title

**HiT-MAC**
Hierarchical Twin-Actor Multi-Agent Coordination
for Directional Sensor Networks

*Algorithm Implementation Overview*
*Codebase: DSN_NA2Q*

---

## Slide 2 — Problem: Target Tracking in Sensor Networks

**Directional Sensor Network (DSN) Target Coverage**

- **Sensors**: PTZ cameras fixed at grid positions
- **Targets**: Multiple moving agents on the grid
- **Goal**: Maximize the fraction of targets tracked at each timestep

**Challenge**:
- Each sensor has a limited field-of-view (angle + range)
- Targets move — sensors must coordinate rotations
- Naive greedy assignment fails: sensors overlap or leave gaps

**Metric**: `coverage_rate` = (targets tracked) / (total targets)

---

## Slide 3 — HiT-MAC Algorithm Overview

**Hierarchical Decomposition** splits the problem into two sub-problems:

```
HIGH LEVEL: WHO tracks WHAT?
  └─ Coordinator assigns sensors → targets

LOW LEVEL: HOW does each sensor orient?
  └─ Executor decides: turn left / stay / turn right
```

Both levels use **A3C (Asynchronous Advantage Actor-Critic)**

> Code entry point: `hitmac/main.py`
> Core models: `hitmac/models.py`

---

## Slide 4 — Two-Level Hierarchy Architecture

```
┌──────────────────────────────────────────┐
│         COORDINATOR (A3C_Multi)          │
│  Input: all sensors × all targets        │
│  Output: binary assignments [S × T]      │
│  "Sensor 3 should track Target 1"        │
└─────────────────┬────────────────────────┘
                  │ assignment matrix
        ┌─────────▼──────────┐
        │                    │
  ┌─────▼──────┐     ┌───────▼─────┐
  │ Executor 1 │     │ Executor N  │
  │ (A3C_Sngl) │ ... │ (A3C_Sngl)  │
  │ turn ±θ   │     │ turn ±θ    │
  └────────────┘     └─────────────┘
```

- **Coordinator**: 1 shared model for all agent assignments
- **Executor**: 1 shared model, run once per sensor
- Code: `hitmac/models.py` — `A3C_Multi` (line 631), `A3C_Single` (line 516)

---

## Slide 5 — Executor: A3C_Single

**Role**: Given "sensor i is assigned to these targets", pick an action.

**Architecture**:
```
Observation [n_targets × 4]        (sensor_id, target_id, distance, angle)
    ↓
AttentionLayer                      (focus on important targets)
    ↓  features [128-dim]
    ├→ PolicyNet (Actor)  → softmax → action probabilities [3]
    └→ ValueNet  (Critic) → scalar value estimate
```

**Actions**:
- `0` = Turn Left
- `1` = Stay
- `2` = Turn Right

**Exploration**: NoisyLinear layers — no epsilon-greedy needed

> `hitmac/models.py:558–624`

---

## Slide 6 — Coordinator: A3C_Multi

**Role**: Assign sensors to targets to maximize global coverage.

**Architecture**:
```
Observation [n_agents × n_targets × 4]
    ↓
EncodeLinear (MLP)                  (per-target feature extraction)
    ↓
AttentionLayer                      (aggregate across agents)
    ↓  features [128-dim]
    ├→ PolicyNet (Actor)  → sigmoid → assignment probs [S × T]
    └→ ValueNet / AMCValueNet (Critic) → value estimate
```

**Optional**: `AMCValueNet` uses **Shapley values** for fair credit assignment
- Tracks marginal contribution of each agent to the coalition

> `hitmac/models.py:683–771` (forward), `241–361` (Shapley critic)

---

## Slide 7 — Perception Modules

### AttentionLayer (`hitmac/perception.py:312–403`)
Scaled dot-product attention:
```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V
```
Lets each sensor focus on relevant targets.

### NoisyLinear (`hitmac/perception.py:73–160`)
Learned noise for exploration:
```
y = (W + σ_w ⊙ ε_w) x + (b + σ_b ⊙ ε_b)
```
Network adapts its own exploration level — no ε-greedy schedule.

### BiRNN (`hitmac/perception.py:167–256`)
Bidirectional RNN for sequential target encoding.
Output dim = 2× hidden size.

---

## Slide 8 — Training: A3C Parallelization

**Asynchronous Advantage Actor-Critic (A3C)**

```
           Shared Model (CPU shared memory)
              ↑ gradient sync ↑
   ┌──────────┬──────────┬──────────┐
   │ Worker 0 │ Worker 1 │ Worker N │   ← mp.Process
   │  Env 0   │  Env 1   │  Env N   │
   └──────────┴──────────┴──────────┘
                                         ← Test Process (separate)
```

**Each worker loop** (`hitmac/train.py:61–205`):
1. **Sync** — load shared model weights
2. **Collect** — run 20 environment steps
3. **Compute** — calculate gradients with GAE
4. **Update** — push gradients to shared model

No experience replay needed — workers explore in parallel.

> Workers = `N_CPUs - 2` (scenario 1), optimizer: SharedAdam

---

## Slide 9 — Loss Functions & GAE

**Generalized Advantage Estimation** (`hitmac/player.py:235–293`):
```
δₜ  = rₜ + γ · V(sₜ₊₁) - V(sₜ)          (TD error)
Aₜ  = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ...   (advantage)
```
Parameters: `γ = 0.99`, `λ (tau) = 1.0`

**Total Loss**:
```
L = L_policy + 0.5 · L_value

L_policy = -log π(aₜ|sₜ) · Aₜ  -  β · H(π)
L_value  = 0.5 · (Gₜ - V(sₜ))²
```
- `β = 0.01` (entropy coefficient — encourages exploration)
- Entropy bonus `H(π)` prevents premature convergence

---

## Slide 10 — Shapley Value Credit Assignment

**Problem**: In cooperative MARL, how to fairly attribute team reward to individual agents?

**Solution**: Approximate Shapley values via marginal contributions:
```
φᵢ = E[ V(S ∪ {i}) - V(S) ]   over all coalitions S
```

**AMCValueNet** (`hitmac/models.py:241–361`):
- Processes agents sequentially
- Tracks cumulative coalition value
- Computes each agent's marginal contribution
- Prevents the **free-rider problem** (an agent contributing nothing still getting credit)

> Enabled when model name contains `"shap"`, e.g. `multi-att-shap`

---

## Slide 11 — Environment: DSNEnv

**File**: `environments/environment.py`

| Property | Value |
|----------|-------|
| Grid size | 3×3 to 10×10 |
| Sensor actions | turn left / stay / turn right |
| Target motion | random velocity (0.3–0.7 units/step) |
| Observation | `[sensor_id, target_id, distance, angle]` per target |
| Reward | `coverage_rate` + threshold bonuses |

**Reward bonuses**:
- `+0.1` at 50% coverage
- `+0.2` at 80% coverage
- `+0.3` at 100% coverage

**Scenarios**:

| # | Sensors | Targets | Max Steps |
|---|---------|---------|-----------|
| 1 | 5 | 6 | 3M |
| 2 | 50 | 60 | 5M |
| 3 | 15 | 20 | 4M |

---

## Slide 12 — Key Hyperparameters

| Parameter | Value | Role |
|-----------|-------|------|
| `lr` | 0.0003–0.0005 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `tau` | 1.0 | GAE decay |
| `entropy` | 0.005–0.01 | Exploration bonus |
| `num_steps` | 20 | Trajectory length per update |
| `lstm_out` | 128–256 | Attention output dim |
| `workers` | N_CPUs - 2 | Parallel training workers |
| `norm_reward` | True | Online reward normalization |

> Presets in `config.py:243–312`

---

## Slide 13 — Checkpoint & Evaluation

**Saved Files** (`hitmac/checkpoints/`):
- `best.pt` — highest reward checkpoint
- `latest.pt` — most recent checkpoint
- `training_history.npz` — rewards, coverage, episode lengths

**Evaluation** (`hitmac/test.py`):
- Runs 10 evaluation episodes
- Metrics: mean reward, mean coverage rate, FPS

**Commands**:
```bash
# Train
python main.py -a hitmac --mode train --scenario 1

# Resume
python main.py -a hitmac --mode train --scenario 1 --resume

# Test
python main.py -a hitmac --mode test --scenario 1
```

---

## Slide 14 — Summary: Why HiT-MAC Works

| Feature | Benefit |
|---------|---------|
| Hierarchical decomposition | Reduces action space; separates assignment from control |
| Attention mechanisms | Each agent focuses on relevant targets |
| A3C parallelism | Fast, sample-efficient training without replay buffer |
| NoisyNet exploration | Self-adapting exploration without ε schedule |
| Shapley credit assignment | Fair reward distribution, prevents free-rider |
| GAE advantage estimation | Low-variance gradient estimates |

---

## Slide 15 — Code Map (Quick Reference)

```
DSN_NA2Q/
├── hitmac/
│   ├── main.py          Entry point, argument parsing
│   ├── models.py        A3C_Single (line 516), A3C_Multi (line 631)
│   ├── perception.py    NoisyLinear (73), BiRNN (167), Attention (312)
│   ├── player.py        Env interaction + GAE loss (235)
│   ├── train.py         A3C worker process (61)
│   ├── test.py          Evaluation + checkpoint save (209)
│   └── shared_optim.py  SharedAdam / SharedRMSprop
├── environments/
│   └── environment.py   DSNEnv: reset (371), step (410)
└── config.py            Scenario hyperparameter presets (243)
```
