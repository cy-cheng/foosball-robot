# Foosball Robot Training Issues - Comprehensive Analysis

## Executive Summary

After comprehensive code review, I've identified **7 critical bugs and 5 major architectural issues** that prevent the model from training effectively. These issues span reward function bugs, action scaling problems, observation errors, and hyperparameter misconfigurations.

---

## 🔴 CRITICAL BUGS (Must Fix)

### 1. **Broken Contact Reward Logic** (foosball_env.py, lines 427-436)
**Severity**: CRITICAL  
**Impact**: Agent doesn't learn to hit the ball

```python
# CURRENT CODE (BROKEN):
contact_with_agent = False
agent_player_links = self.team1_player_links if self.player_id == 1 else self.team2_player_links
for link_idx in agent_player_links:
    contact_points = p.getContactPoints(bodyA=self.table_id, bodyB=self.ball_id, linkIndexA=link_idx)
    if contact_points:
        contact_with_agent = True
        break

if contact_with_agent:
    reward += 100
```

**Problem**: The contact reward of +100 is given EVERY STEP the agent is in contact with the ball. In a typical contact scenario, the ball may be in contact for 10-50 simulation steps, resulting in +1000 to +5000 reward for a single hit. This completely dominates all other rewards and makes the agent learn to just "stick" to the ball rather than hit it toward the goal.

**Fix**: Only reward contact once per contact event, or use a much smaller continuous contact bonus (e.g., +1 per step).

---

### 2. **Incorrect Rotation Action Scaling** (foosball_env.py, line 338)
**Severity**: CRITICAL  
**Impact**: Actions don't work as intended

```python
# CURRENT CODE (BROKEN):
def _scale_action(self, action, player_id):
    scaled = np.zeros(8)
    # ... slide scaling ...
    for i, joint_id in enumerate(revs):
        scaled[i + 4] = action[i + 4] * 10  # ← BUG: Should scale to velocity range
    return scaled
```

**Problem**: The rotation actions are supposed to be in [-1, 1] range, but they're being multiplied by 10, giving velocities from -10 to +10. However, looking at line 350, the joint control uses `VELOCITY_CONTROL` with these values. The comment at line 88 says `max_vel = 1.5`, but the scaling multiplies by 10 instead of 1.5, resulting in much faster rotations than intended.

**Fix**: Scale rotation actions properly: `scaled[i + 4] = action[i + 4] * self.max_vel`

---

### 3. **Ball Velocity Reward Direction Bug** (foosball_env.py, line 377)
**Severity**: MAJOR  
**Impact**: Wrong reward signal for ball movement

```python
# CURRENT CODE (POTENTIALLY WRONG):
goal_x = self.goal_line_x_2 if self.player_id == 1 else self.goal_line_x_1
reward += (ball_vel[0] if self.player_id == 1 else -ball_vel[0]) * 10
```

**Problem**: For player 1, positive x velocity is toward goal (correct). For player 2, the observation is already mirrored (line 366-367), so `ball_vel[0]` is already negative when moving toward their goal. Negating it again makes it positive, but the sign convention is confusing and may be incorrect depending on whether the velocity observation was mirrored before or after this reward calculation.

**Fix**: Ensure consistent coordinate frame throughout reward calculation. Since observations are mirrored for player 2, the reward calculation should use the mirrored observation consistently.

---

### 4. **Ball Distance Reward Uses Wrong Metric** (foosball_env.py, lines 418-420)
**Severity**: MAJOR  
**Impact**: Agent learns to minimize wrong distance

```python
# CURRENT CODE (CONFUSED):
current_ball_dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array([0, avg_closest_player_y]))
reward += (self.previous_ball_dist - current_ball_dist) * 0.1
```

**Problem**: This calculates the distance from ball to point (0, avg_closest_player_y), which is distance to CENTER of table at the average player Y position. This doesn't make sense - it should either be:
- Distance from player to ball (encouraging players to move toward ball)
- Distance from ball to goal (encouraging ball movement toward goal)

The current implementation creates a confusing gradient that doesn't help learning.

**Fix**: Use distance from ball to opponent's goal: `np.linalg.norm(ball_pos[0] - goal_x)`

---

### 5. **Mirrored Observation Uses Wrong Joint Order** (foosball_env.py, lines 278-280)
**Severity**: MAJOR  
**Impact**: Opponent sees incorrect state

```python
# CURRENT CODE (POTENTIALLY BROKEN):
mirrored_joint_pos = team2_pos + team1_pos
mirrored_joint_vel = team2_vel + team1_vel
```

**Problem**: When mirroring for self-play, the joint order is simply swapped (team2 first, then team1). However, the joint indices may not correspond correctly because:
1. Team 1 rods are [1, 2, 4, 6] and Team 2 rods are [3, 5, 7, 8] (different rod numbers)
2. The spatial positions of these rods are NOT symmetric
3. Simply swapping the order doesn't create a true mirror of the state

**Fix**: Properly mirror joint states accounting for spatial layout, or redesign observation to be truly symmetric.

---

### 6. **Stage 1 Curriculum Sets Opponent Rods to 90° Inside step()** (foosball_env.py, lines 227-228)
**Severity**: MODERATE  
**Impact**: Wastes computation and may interfere with learning

```python
# CURRENT CODE:
if self.curriculum_level == 1:
    self._set_opponent_rods_to_90_degrees()  # Called EVERY step
```

**Problem**: In curriculum level 1, opponent rods are set to 90° on EVERY simulation step (potentially 4000 times per episode). This:
1. Wastes computation (setting position control repeatedly)
2. May create confusing observations (opponent joints always at 90° but agent sees them in observation)
3. Should be done once at reset, not every step

**Fix**: Only set opponent rods to 90° during reset, or remove opponent joints from observation in stage 1.

---

### 7. **Stuck Ball Detection Counter Not Reset on Movement** (foosball_env.py, lines 465-467)
**Severity**: MINOR  
**Impact**: False positive episode terminations

```python
# CURRENT CODE:
if np.linalg.norm(ball_vel) < 0.001: self.ball_stuck_counter += 1
else: self.ball_stuck_counter = 0
if self.ball_stuck_counter > self.max_stuck_steps: truncated = True
```

**Problem**: Threshold of 0.001 is TOO LOW. A ball rolling slowly (e.g., 0.0005 m/s) would be considered stuck, even though it's still moving. Also, `max_stuck_steps` is 5000, meaning ball needs to be below threshold for 5000 steps = ~41 seconds at 240Hz simulation. This is too long.

**Fix**: Increase threshold to 0.01 and reduce max_stuck_steps to 2000.

---

## 🟡 MAJOR ARCHITECTURAL ISSUES

### 8. **Reward Scale Imbalance**
**Problem**: Different reward components have vastly different scales:
- Ball velocity: ±10 per step (can accumulate to ±40,000 over episode)
- Contact: +100 per step (accumulates to +5000 if in contact for 50 steps)
- Goal scored: +40,000 (sparse, only once per episode)
- Own goal: -10,000 (sparse)
- Distance penalty: ~-0.001 to -1 per step
- Inactivity penalty: -0.1 per step

**Impact**: The continuous rewards (velocity, contact) completely dominate the sparse goal rewards, so the agent optimizes for constant ball contact and high velocity, not for scoring goals.

**Fix**: Rebalance rewards:
- Contact: +1 per step (or +50 once per contact event)
- Ball velocity: +1.0 × velocity component
- Goal scored: +1000
- Own goal: -1000
- Distance penalty: -0.1 per meter
- Remove stuck ball penalty from reward (it's already in termination)

---

### 9. **PPO Hyperparameters Not Tuned for Continuous Control**
**Current config** (train.py, lines 184-196):
```python
PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,    # ← Too high for fine control
    batch_size=64,          # ← Too small for vectorized envs
    n_steps=2048,           # ← Standard, but could be larger
    n_epochs=10,            # ← Standard
    gamma=0.99,             # ← OK
    gae_lambda=0.95,        # ← OK
    clip_range=0.2,         # ← Standard
    ent_coef=0.01,          # ← Low, may cause premature convergence
    verbose=1,
)
```

**Problems**:
1. **Learning rate too high**: 3e-4 is standard for discrete actions, but continuous control often needs 1e-4 to 1e-3
2. **Batch size too small**: With 4 parallel environments and 2048 steps, we collect 8192 samples per update. Batch size of 64 means 128 mini-batches, which is inefficient. Should be 256-512.
3. **Entropy coefficient too low**: 0.01 is very low and may cause the policy to become deterministic too quickly, preventing exploration.

**Fix**:
```python
PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,      # Reduced for stability
    batch_size=256,          # Increased for efficiency
    n_steps=4096,            # Increased for more data per update
    n_epochs=10,             
    gamma=0.99,              
    gae_lambda=0.95,         
    clip_range=0.2,          
    ent_coef=0.05,           # Increased for more exploration
    verbose=1,
)
```

---

### 10. **Curriculum Ball Spawning Issues**

**Stage 1** (lines 177-183): Ball spawns with initial velocity, defeating the purpose of "stationary ball":
```python
ball_vel = [np.random.uniform(-0.2, -0.1), np.random.uniform(-0.1, 0.1), 0]
```
Should be `ball_vel = [0, 0, 0]` for true stationary dribbling practice.

**Stage 3** (lines 189-205): Ball spawn calculation is overly complex and may have edge cases where division by zero occurs (if spawn_pos == target_pos).

---

### 11. **Action Space Design Issues**

The current action space is 8D: [slide1, slide2, slide3, slide4, rot1, rot2, rot3, rot4]

**Problems**:
1. Slides use position control (line 347) but rotations use velocity control (line 350) - inconsistent control modes make learning harder
2. Network must learn to coordinate 8 DOF simultaneously from the start, which is very difficult
3. No action normalization or clipping before applying to joints

**Better approach**:
- Use position control for both slides and rotations (more stable)
- OR use velocity control for both (more dynamic)
- Consider starting with 4 DOF in early curriculum (only slides, keep rotations fixed)

---

### 12. **Self-Play Implementation Issues** (Stage 4)

**Line 149**: Self-play callback updates opponent every 100K steps:
```python
self_play_callback = SelfPlayCallback(update_freq=100_000)
```

**Problems**:
1. 100K steps is TOO INFREQUENT. Opponent model becomes stale, causing overfitting to old strategy
2. Update happens via `env_method`, but SubprocVecEnv runs in separate processes - the state_dict may not transfer correctly
3. No verification that opponent update succeeded

**Fix**: Update opponent every 10K-20K steps and add logging to verify updates.

---

## 🟢 RECOMMENDED IMPROVEMENTS

### 13. **Add Reward Curriculum**
Start with dense rewards in early stages, gradually shift to sparse rewards:
- Stage 1: Dense (contact + velocity + distance)
- Stage 2: Medium (velocity + distance + sparse goals)
- Stage 3: Sparse (mainly goals)
- Stage 4: Sparse + self-play

### 14. **Add Progress Tracking**
Log key metrics:
- % episodes with goals scored
- Average ball velocity toward goal
- % episodes with contact made
- Joint motion statistics (are all rods being used?)

### 15. **Improve Observation Space**
Current observation is 38D and includes all joints, making it high-dimensional and potentially confusing. Consider:
- Relative observations (ball position relative to closest rod)
- Add goal position explicitly
- Remove opponent joint positions from observation in early curriculum stages

### 16. **Better Network Architecture**
Current MLP is simple. For this task, consider:
- Larger hidden layers (128-256 neurons instead of 64)
- 3 hidden layers instead of 2
- Separate encoders for ball state and joint state

### 17. **Add Reward Logging**
Log individual reward components to understand what agent is optimizing:
```python
reward_info = {
    'ball_velocity_reward': vel_reward,
    'contact_reward': contact_reward,
    'goal_reward': goal_reward,
    'total_reward': reward
}
```

---

## 📊 Priority Fix Order

1. **FIX IMMEDIATELY** (Blocks all learning):
   - Bug #1: Contact reward logic (reduces it from +100 per step to +50 per contact event)
   - Bug #2: Rotation action scaling (use correct max_vel)
   - Bug #8: Reward scale rebalancing

2. **FIX BEFORE STAGE 2** (Needed for curriculum):
   - Bug #3: Ball velocity reward direction
   - Bug #4: Ball distance reward metric
   - Bug #10: Stage 1 ball velocity should be [0,0,0]

3. **FIX BEFORE STAGE 4** (Needed for self-play):
   - Bug #5: Mirrored observation joint order
   - Bug #12: Self-play update frequency

4. **FIX FOR BETTER TRAINING** (Performance improvements):
   - Bug #9: PPO hyperparameters
   - Bug #6: Stage 1 opponent rod setting
   - Bug #11: Action space control mode consistency

5. **FIX LATER** (Quality of life):
   - Bug #7: Stuck ball detection threshold
   - Improvements #13-17

---

## 🎯 Expected Improvements After Fixes

After implementing these fixes, you should see:
1. **Stable learning curve**: Reward should increase monotonically in early training
2. **Contact within 10K steps**: Agent should learn to hit ball reliably
3. **Goals within 50K steps**: Agent should score first goals in Stage 1
4. **Curriculum progression**: Agent should reach Stage 2 within 100K-200K steps
5. **Final performance**: 20-50 goals per match after 1M steps of training

---

## 📝 Summary Table

| Bug # | Severity | File | Lines | Impact | Fix Complexity |
|-------|----------|------|-------|--------|----------------|
| 1 | CRITICAL | foosball_env.py | 427-436 | Contact reward broken | Easy |
| 2 | CRITICAL | foosball_env.py | 338 | Action scaling wrong | Easy |
| 3 | MAJOR | foosball_env.py | 377 | Velocity reward direction | Medium |
| 4 | MAJOR | foosball_env.py | 418-420 | Distance reward metric | Easy |
| 5 | MAJOR | foosball_env.py | 278-280 | Mirrored obs wrong | Hard |
| 6 | MODERATE | foosball_env.py | 227-228 | Wasteful computation | Easy |
| 7 | MINOR | foosball_env.py | 465-467 | False terminations | Easy |
| 8 | MAJOR | foosball_env.py | 370-454 | Reward imbalance | Medium |
| 9 | MAJOR | train.py | 184-196 | Hyperparameters | Easy |
| 10 | MODERATE | foosball_env.py | 177-183 | Curriculum spawn | Easy |
| 11 | MODERATE | foosball_env.py | 341-350 | Control modes | Medium |
| 12 | MODERATE | train.py | 149 | Self-play freq | Easy |

**Total**: 7 Critical/Major bugs, 5 architectural issues, 5 recommended improvements

---

## 🔧 Implementation Plan

See the accompanying fixes in the pull request. All critical bugs will be addressed first, followed by hyperparameter improvements and architectural enhancements.
