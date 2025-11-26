# 🎮 Foosball RL Training System

Complete end-to-end training system for two-agent foosball using symmetric RL and curriculum learning.

## 📦 Installation

```bash
uv pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Train a Model (Symmetric Single Agent)

Train a single policy that will be mirrored for both teams:

```bash
uv run train.py --mode train --steps 1000000 --level 1 --num-envs 1
```

**Key Arguments:**
- `--steps`: Total training timesteps (default: 1M)
- `--level`: Starting curriculum level 1-4 (default: 1)
  - Level 1: Dribble (ball stationary)
  - Level 2: Pass (ball rolling at midfield)
  - Level 3: Defend (ball shot at goal from opponent)
  - Level 4: Full Game (random play)
- `--num-envs`: Parallel environments for faster training
- `--lr`: Learning rate (default: 3e-4)
- `--batch-size`: Mini-batch size (default: 64)
- `--model`: Custom model name (default: "foosball_model")

### 2. Evaluate the Model

Run symmetric two-agent matches:

```bash
uv run test.py --model saves/foosball_model_final.zip --episodes 10
```

**Key Arguments:**
- `--model`: Path to trained model
- `--episodes`: Number of test episodes
- `--no-render`: Disable GUI visualization

### 3. Train on Specific Level Only

Start training at a specific curriculum level:

```bash
uv run train.py --mode train --level 3 --steps 500000
```

## 📊 Architecture

### Environment (`foosball_env.py`)

**Symmetric Observation:**
- Ball position (3D), velocity (3D)
- All 16 joint positions + velocities
- For Player 2, X coordinates are mirrored (enables symmetric training)

**Action Space:**
- 8 continuous values: 4 slide joints + 4 rotation joints
- Scaled to joint limits [-1, 1] → actual ranges

**Curriculum Levels:**

| Level | Scenario | Agent Focus |
|-------|----------|------------|
| 1 | Ball stationary in front | Dribbling / Basic control |
| 2 | Ball rolling from midfield | Intercepting / Passing |
| 3 | Fast shot from opponent goal | Defending / Blocking |
| 4 | Random play anywhere | Full game strategy |

**Auto-Progression:** Advances to next level after 10 goals

### Reward Structure

**Dense Rewards (every step):**
- Ball velocity towards opponent: +0.1 × velocity[0]
- Distance to goal: -0.01 × distance (encourages shooting)
- Rod extension: +0.1 × avg_extension (encourages active play)

**Sparse Rewards (terminal):**
- Goal scored: +100
- Own goal: -50

### Training Strategy

**Symmetric Single-Agent Training:**
1. Train ONE policy π on curriculum (1M steps)
2. Deploy π to both Team 1 and Team 2
3. For Team 2: Mirror all X-axis actions/observations
4. Result: Balanced, symmetric gameplay

**Benefits:**
- 50% faster training (half the agents)
- Automatically balanced
- Learns universal foosball fundamentals

## 📈 Expected Learning Curve

After ~500K steps:
- Level 1: Agent learns to hit ball forward consistently
- Level 2: Learns to intercept mid-field balls
- Level 3: Defensive positioning improves
- Level 4: Full game strategy emerges (~30-50 goals per match)

## 📁 File Structure

```
foosball_robot/
├── foosball_env.py          # Two-agent symmetric environment
├── train.py                 # Training script with curriculum
├── test.py                  # Evaluation script
├── complete_test.py         # Manual testing (legacy)
├── foosball.urdf            # Table/ball physics model
├── requirements.txt         # Python dependencies
│
├── saves/                   # Model checkpoints
│   ├── foosball_model_ckpt_50000_steps.zip
│   ├── foosball_model_ckpt_100000_steps.zip
│   └── foosball_model_final.zip
│
└── logs/                    # TensorBoard logs
    └── foosball_model_20250125_143000/
        └── events.*
```

## 🎓 Training Tips

### Quick Experiment (5 minutes)
```bash
uv run train.py --steps 100000 --level 1 --num-envs 4
```

### Production Training (1-2 hours)
```bash
uv run train.py --steps 1000000 --level 1 --num-envs 8 --lr 3e-4
```

### Monitor Progress with TensorBoard
```bash
tensorboard --logdir logs/
# Navigate to http://localhost:6006
```

### Resume Training
```bash
# Modify train.py to load checkpoint instead of creating new model
model = PPO.load("saves/foosball_model_ckpt_500000_steps")
model.learn(total_timesteps=500000, callback=...)
```

## 🔍 Debugging

**Verify Environment:**
```bash
python foosball_env.py
```
This tests joint parsing and basic simulation.

**Manual Control:**
```bash
# In train.py, set debug_mode=True
# Use sliders in GUI to control rods manually
```

**View Model Architecture:**
```python
from stable_baselines3 import PPO
model = PPO.load("saves/foosball_model_final")
print(model.policy)  # Print network architecture
```

## 🤝 Two-Agent Scenarios

### Scenario 1: Symmetric Training (Recommended)
- Single policy trained
- Mirrored for both teams
- **Use:** `uv run train.py` (default)

### Scenario 2: Self-Play (Advanced)
- Train two separate policies
- Periodically swap
- (Future enhancement)

### Scenario 3: Mixed Training
- Train Team 1 vs fixed hardcoded Team 2
- Currently supported via `player_id=1` parameter

## 🐛 Common Issues

**Issue:** Model not saving
- Check `saves/` directory exists
- Ensure write permissions

**Issue:** Training is slow
- Increase `--num-envs` for parallel simulation
- Reduce `--batch-size` for faster updates
- Set `--n-steps` lower (trades for more updates)

**Issue:** Agent not learning (flat reward curve)
- Try earlier curriculum level `--level 1`
- Increase `--lr` slightly (e.g., 5e-4)
- Verify observations are changing with `print(obs)`

**Issue:** Joint positions not parsing correctly
- Check URDF joint naming: must contain "1" or "2" for team
- Must contain "slide" or "rev"/"rotate" for type
- Verify with `python foosball_env.py` debug output

## 📚 References

- **Stable Baselines3:** https://stable-baselines3.readthedocs.io/
- **Gymnasium:** https://gymnasium.farama.org/
- **PyBullet:** https://pybullet.org/
- **PPO Paper:** https://arxiv.org/abs/1707.06347

## 🚀 Next Steps

1. ✅ Single-agent symmetric training
2. 🔜 Self-play adversarial training
3. 🔜 Multi-agent independent training + joint optimization
4. 🔜 Imitation learning from human demonstrations
5. 🔜 Video recording of matches
6. 🔜 Benchmark against hardcoded strategies

## 📝 License

See LICENSE file.
