# Foosball RL Training Plan

## Phase 1: Environment Upgrade (New `foosball_env.py`)
### Key Changes:
1. **Two-Agent Symmetric Environment**
   - Gym Env where both agents get identical observation space and action space
   - Support training either Team 1 OR Team 2 via a `player_id` parameter
   - Actions for the other team come from RL policy (not hardcoded bot)
   - Automatic action mirroring: Team 2 sees negated X positions/velocities (symmetry)

2. **Richer Reward Structure** (solve sparse reward problem)
   - **Sparse rewards**:
     - +100 for scoring own goal
     - -50 for conceding goal (own goal)
   - **Dense rewards**:
     - Ball velocity forward (toward opponent goal): `+0.1 * (ball_vel.x toward opponent)`
     - Ball distance to opponent goal: `-0.01 * distance` (encourage shooting)
     - Successful ball interception: `+5` (when ball velocity reverses near rods)
     - Rod extension bonus: `+1` per step if rod extended (encourages active play)

3. **Curriculum Levels** (implemented in reset)
   - **Level 1 - Dribble**: Ball stationary in front of own offensive rod
   - **Level 2 - Pass**: Ball at midfield moving slowly toward own rods
   - **Level 3 - Defend**: Ball on opponent side shooting fast toward own goal
   - **Level 4 - Full Game**: Random ball position/velocity anywhere on table
   - Smooth curriculum progression: After N goals, advance level

4. **Ball Stuck Detection**
   - If ball velocity < 0.001 for 1500 steps → truncate episode (return truncated=True)
   - Prevents agent from learning to stall

## Phase 2: Training Strategy (New `train.py`)

### Option A: Symmetric Single-Agent Training (Recommended)
**Rationale**: Game is symmetric. Train ONE policy, mirror it for opponent.

**Steps**:
1. Create `FoosballEnv(player_id=1)` - only Team 1 can be trained
2. Train single policy `π` using PPO/A3C for ~1M steps on curriculum
3. At test time, apply policy to both teams (mirror actions for Team 2)
4. Result: Both teams play identically, creates balanced gameplay

**Pros**: 
- 2x fewer training steps
- Simpler code
- Automatically balanced (identical agents)

**Cons**: 
- May converge to suboptimal equilibrium (e.g., both playing passively)

### Option B: Self-Play / Adversarial Training (Advanced)
**If symmetric training creates boring play**:

1. **Stage 1**: Train single policy on curriculum (1M steps)
2. **Stage 2**: Freeze π_main, train new policy π_opponent against it (500K steps)
3. **Stage 3**: Periodically swap policies and continue (prevents exploitation)
4. Repeat stages 2-3 until both policies converge

**Pros**: 
- Better emergent behaviors
- More diverse play

**Cons**: 
- 3-4x training time
- More complex logic

### Option C: Hybrid (Recommended for robustness)
1. Train single policy `π` on full curriculum (1M steps)
2. Create two copies: `π_A` and `π_B` (both initialized from π)
3. Self-play training for 500K steps (optional, if needed)
4. Deploy both

---

## Phase 3: Implementation Checklist

### `foosball_env.py` Changes:
- [ ] Add `player_id` parameter (1 or 2) to env
- [ ] Implement observation mirroring for Team 2
- [ ] Implement action mirroring for Team 2
- [ ] Add dense reward shaping (ball velocity, distance, interception)
- [ ] Curriculum auto-advance based on goal count
- [ ] Fix action scaling for slide/revolute joints
- [ ] Add ball stuck counter + truncation logic

### `train.py` Implementation:
- [ ] Load `FoosballEnv(player_id=1)`
- [ ] Initialize PPO agent
- [ ] Curriculum callback (advance level every N goals)
- [ ] Training loop with checkpoints every 50K steps
- [ ] Evaluation loop (test both symmetric agents)
- [ ] Logging: goals scored, episode length, reward curves
- [ ] Save best model to `saves/`

### `test.py` (for evaluation):
- [ ] Load trained policy
- [ ] Run symmetric two-agent match
- [ ] Log scores and video (optional)

---

## Hyperparameters (Starting Point)

```python
# PPO
learning_rate = 3e-4
batch_size = 64
n_steps = 2048
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2

# Curriculum
curriculum_change_freq = 5000  # steps
goals_per_level = 10

# Environment
episode_max_steps = 10000
max_stuck_steps = 1500
```

---

## Expected Results

**After Phase 1 training (1M steps, single policy)**:
- Agent learns to intercept ball at midfield
- Simple offensive patterns emerge
- Scores ~30-50 goals in 1000 episode match

**After Phase 2+ (self-play, if needed)**:
- More sophisticated offensive strategies
- Better defensive positioning
- Scores ~50-100+ goals

---

## File Structure
```
foosball_robot/
├── foosball_env.py       # New: Two-agent symmetric env
├── train.py              # New: Training script with curriculum
├── test.py               # New: Evaluation/testing
├── complete_test.py      # Keep for reference (manual testing)
├── saves/                # Checkpoints go here
│   ├── best_model.zip
│   └── curriculum_level_*.zip
└── logs/                 # TensorBoard logs
```

---

## Key Decisions

1. **Why symmetric training?** 
   - Game is perfectly symmetric
   - Reduces training by 50%
   - Guarantees balanced play

2. **Why curriculum?**
   - Sparse reward makes RL hard
   - Curriculum provides learning signal at each stage
   - Smooth progression → faster convergence

3. **Why dense rewards?**
   - Ball velocity forward encourages offensive play
   - Distance reward prevents passive behavior
   - Rod extension bonus encourages active rod usage

4. **Why PPO over other algorithms?**
   - Stable and sample-efficient
   - Works well with curriculum learning
   - Easy to tune

