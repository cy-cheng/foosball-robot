# 🎮 Foosball RL - Quick Reference

## Installation (One-time)
```bash
uv pip install -r requirements.txt
```

## Common Commands

### 1. Verify Setup
```bash
python foosball_env.py
# Should print: Observation shape: (38,), Action space: Box(...)
```

### 2. Train Models

**Quick test (5 min)**
```bash
uv run train.py --steps 50000 --level 1 --num-envs 2 --no-render
```

**Short training (30 min)**
```bash
uv run train.py --steps 200000 --level 1 --num-envs 4
```

**Full training (2-4 hours)**
```bash
uv run train.py --steps 1000000 --level 1 --num-envs 8
```

**Resume training from checkpoint**
```bash
# Edit train.py: model = PPO.load("saves/foosball_model_ckpt_500000_steps")
uv run train.py --steps 500000 --level 2
```

### 3. Evaluate Models

**Quick eval (no render)**
```bash
uv run test.py --model saves/foosball_model_final.zip --episodes 3 --no-render
```

**Full eval with visualization**
```bash
uv run test.py --model saves/foosball_model_final.zip --episodes 10
```

### 4. Monitor Training

**TensorBoard (in separate terminal)**
```bash
tensorboard --logdir logs/
# Open: http://localhost:6006
```

### 5. Debug

**Check joint names**
```python
import pybullet as p
p.connect(p.DIRECT)
table = p.loadURDF("foosball.urdf")
for i in range(p.getNumJoints(table)):
    print(p.getJointInfo(table, i)[1].decode())
```

**Test environment manually**
```python
from foosball_env import FoosballEnv
env = FoosballEnv(render_mode='human', debug_mode=True)
obs, _ = env.reset()
for _ in range(100):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc: break
```

---

## Training Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--steps` | 1M | 10K-10M | Total training timesteps |
| `--level` | 1 | 1-4 | Starting curriculum level |
| `--num-envs` | 1 | 1-16 | Parallel environments (faster) |
| `--lr` | 3e-4 | 1e-5 to 1e-3 | Learning rate (higher = less stable) |
| `--batch-size` | 64 | 32-256 | Batch size (higher = less noise) |
| `--n-steps` | 2048 | 512-8192 | Rollout length (higher = more stable) |

### Tuning Tips

**Agent not learning?**
- ↓ Start at Level 1
- ↑ Increase `--lr` slightly
- ↑ Increase `--num-envs`

**Training too slow?**
- ↑ Increase `--num-envs` (2x speedup per env)
- ↓ Reduce `--batch-size`
- ↓ Reduce `--n-steps`

**Training unstable?**
- ↑ Increase `--batch-size`
- ↑ Increase `--n-steps`
- ↓ Decrease `--lr`

---

## Curriculum Levels Explained

| Level | What Happens | Agent Should Learn | Duration |
|-------|-------------|-------------------|----------|
| 1 | Ball spawns stationary in front | Basic hitting, dribbling | ~200K steps |
| 2 | Ball rolls toward you | Intercepting, timing | ~250K steps |
| 3 | Ball shot at your goal | Defending, blocking | ~250K steps |
| 4 | Random full game | Offensive + defensive strategy | ~300K steps |

**Automatic progression**: After 10 goals in current level, advance to next

---

## File Locations

```
saves/
  foosball_model_ckpt_50000_steps.zip     ← Checkpoint at 50K steps
  foosball_model_ckpt_100000_steps.zip    ← Checkpoint at 100K steps
  foosball_model_final.zip                ← Final trained model

logs/
  foosball_model_YYYYMMDD_HHMMSS/
    events.out.tfevents.*                 ← TensorBoard logs
```

---

## Architecture at a Glance

```
Input (38D):
  - Ball: position (3) + velocity (3)
  - Joints: 16 positions + 16 velocities

MLP Policy (PPO):
  input → hidden (64) → hidden (64) → output (8D)

Output (8D):
  - Rods 1-4: slide command
  - Rods 1-4: rotate command
```

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| ImportError: gymnasium | Missing dependency | `uv pip install -r requirements.txt` |
| Obs shape (14,) not (38,) | Joint parsing failed | Check URDF joint names have "rod_X" |
| Model not learning | Bad curriculum/reward | Start at Level 1, increase --lr |
| Out of memory | Too many parallel envs | Reduce `--num-envs` |
| Training crashes | GUI issues | Add `--no-render` |
| Goals never triggered | Wrong goal lines | Check `goal_line_x_1`, `goal_line_x_2` |

---

## Reward Breakdown

Each step:
- Velocity reward: +0.1 × velocity_toward_goal
- Distance penalty: -0.01 × distance_to_goal  
- Rod usage: +0.1 × avg_rod_extension

Terminal states:
- Goal scored: +100
- Own goal: -50

---

## Expected Performance

```
Time        | Steps  | Level Progress | Typical Goals/Match
------------|--------|----------------|--------------------
0 min       | 0      | 1              | 0-2
5 min       | 10K    | 1              | 1-5
15 min      | 50K    | 1-2            | 3-10
1 hour      | 200K   | 2-3            | 10-20
2 hours     | 500K   | 3-4            | 20-35
4 hours     | 1M     | 4              | 30-50+
```

---

## Player Mapping

```
Blue Team (Player 1):
  - Rods 1, 2, 3, 4 (left side)
  - Goal on right (+X direction)
  - Observation X mirrored: False

Red Team (Player 2):
  - Rods 5, 6, 7, 8 (right side)
  - Goal on left (-X direction)
  - Observation X mirrored: True (symmetric training)
```

---

## Training Phases

```
PHASE 1: SINGLE AGENT TRAINING
─────────────────────────────
Policy π trained on curriculum 1M steps

PHASE 2: DEPLOYMENT
─────────────────────────────
π → Blue team
π (mirrored) → Red team
Result: Symmetric, balanced gameplay

OPTIONAL PHASE 3: SELF-PLAY
─────────────────────────────
π_A vs π_B (both evolve separately)
More diverse strategies emerge
```

---

## Performance Metrics to Track

- **Episode Reward**: Should increase (goal is >0)
- **Episode Length**: Should increase early, then stabilize (~500-10000)
- **Goals Scored**: Should increase significantly after Level 2
- **Curriculum Level**: Should progress 1→2→3→4
- **Value Loss**: Should decrease over time
- **Policy Loss**: Should be small and stable

---

## Next Steps After Training

1. **Evaluate**
   ```bash
   uv run test.py --model saves/foosball_model_final.zip --episodes 20
   ```

2. **Deploy**
   - Use model for both teams
   - Actions automatically mirrored

3. **Improve (Optional)**
   - Self-play: train separate opponent
   - Curriculum fine-tuning
   - Adversarial training

---

**Quick Start**: `uv run train.py --steps 100000 --level 1 --num-envs 4`

Good luck! 🚀
