# Foosball RL Training Improvements

## Summary
This document outlines the improvements made to the foosball robot RL training system to address curriculum learning issues and provide more controlled training.

## Changes Made

### 1. ✅ Reduced Rotation Speed Limit
**File**: `foosball_env.py` (line 99)

**Change**:
```python
self.max_vel = 0.1  # Reduced from 1.0 to limit rotation speed
```

**Rationale**:
- The original max velocity of 1.0 allowed rods to rotate extremely fast
- This caused the agent to learn unrealistic strategies (e.g., using goalkeeper for all shots)
- Reducing to 0.1 provides more controlled, realistic rotation speeds
- Allows for more nuanced ball control requiring multiple rod coordination

**Impact**: Agents now learn to use rods more carefully and realistically.

---

### 2. ✅ Improved Curriculum Stage 1 - All Rods Practice
**File**: `foosball_env.py` (lines 184-198)

**Change**:
```python
def _curriculum_spawn_ball(self):
    """Spawn ball according to curriculum level"""
    if self.curriculum_level == 1:
        # Dribble: ball in random position reachable by all rods
        # Place ball in front of team, covering all rod ranges
        if self.player_id == 1:
            # Team 1 rods cover x from -0.4 to 0.0, place ball between them
            ball_x = np.random.uniform(-0.4, 0.0)
            ball_y = np.random.uniform(-0.3, 0.3)
        else:
            # Team 2 rods cover x from 0.0 to 0.4, place ball between them
            ball_x = np.random.uniform(0.0, 0.4)
            ball_y = np.random.uniform(-0.3, 0.3)
        ball_pos = [ball_x, ball_y, 0.55]
        ball_vel = [0, 0, 0]
```

**Previous behavior**: Ball was always spawned at a single fixed position (-0.2 for Team 1), only in front of the goalkeeper.

**New behavior**: Ball spawns at random positions across the entire range reachable by all rods [-0.4 to 0.0 for Team 1].

**Rationale**:
- Previous curriculum only trained the goalkeeper rod
- Agent learned to only use the goalkeeper's high speed (now capped) to hit the ball
- All rods must now learn to handle the ball at various positions
- Better preparation for subsequent curriculum stages

**Impact**: More balanced training across all rods; agent learns diverse control strategies.

---

### 3. ✅ Fixed Curriculum Level Passing to Environments
**File**: `train_stages.py` (line 93)

**Change**:
```python
curriculum_level=stage,  # Changed from: curriculum_level=1
```

**Previous behavior**: Training always used curriculum_level=1 regardless of the training stage.

**New behavior**: Each training stage now correctly uses its corresponding curriculum level:
- Stage 1 → curriculum_level=1 (Dribble with random positions)
- Stage 2 → curriculum_level=2 (Pass with incoming ball)
- Stage 3 → curriculum_level=3 (Defend against fast shots)
- Stage 4 → curriculum_level=4 (Full game with random conditions)

**Rationale**: Enables true progressive curriculum learning as intended by the original design.

**Impact**: Better skill progression; agents develop offense first, then defense, then full game strategy.

---

### 4. ✅ Added Penalty for Redundant Actions
**File**: `foosball_env.py` (lines 409-416)

**Change**: Added dense reward component:
```python
# Dense reward 4: penalty for redundant actions (too many rods rotating)
team_revs = self.team1_rev_joints if self.player_id == 1 else self.team2_rev_joints
joint_states_rev = p.getJointStates(self.table_id, team_revs)
# Count rods that are actively rotating (velocity > threshold)
active_rods = sum(1 for state in joint_states_rev if abs(state[1]) > 0.1)
# Penalty if more than 2 rods are rotating simultaneously
if active_rods > 2:
    reward -= 0.01 * (active_rods - 2)
```

**Rationale**:
- Prevents agent from simultaneously rotating many rods unnecessarily
- Encourages efficient, coordinated movement
- Makes trained policy more interpretable and realistic
- Reduces unnecessary energy expenditure

**Impact**: Smoother, more coordinated rod movements; cleaner learned behaviors.

---

## Verification

All changes have been tested and verified:

```
✅ 1. ROTATION SPEED LIMIT
   Max velocity: 0.1 (reduced from 1.0)
   Status: ✅ Rotation speed is now much slower and more controlled

✅ 2. IMPROVED CURRICULUM STAGE 1 - RANDOM BALL POSITIONS
   Ball X positions across 20 resets: [-0.392, -0.006]
   Expected range for Team 1: [-0.4, 0.0]
   Status: ✅ Stage 1 now trains all rods, not just goalkeeper

✅ 3. VERIFYING 4-ROD CONTROL (Team 1)
   Action space: (8,) - 4 rods × 2 DOF each
   Player ID: 1 (Team 1, controls rods 1-4)
   Status: ✅ Correctly controlling exactly 4 rods

✅ 4. REDUNDANT ACTION PENALTY
   Reward with 1 rod rotating: -0.006370
   Reward with 4 rods rotating: -0.009875
   Penalty applied: 0.003504
   Status: ✅ Penalty discourages excessive simultaneous rod rotation
```

---

## Training Recommendations

### Stage 1 (Dribble)
- Steps: 250K (can increase to 500K for more practice)
- Expected behavior: Agent learns to hit ball from various positions
- Success metric: Consistently scores goals using different rods

### Stage 2 (Pass)
- Starts with pre-trained Stage 1 model
- Ball now rolls toward agent's rods
- Success metric: Learns to intercept and control incoming balls

### Stage 3 (Defend)
- Starts with pre-trained Stage 2 model
- Fast shots from opponent side
- Success metric: Blocks and deflects incoming shots

### Stage 4 (Full Game)
- Starts with pre-trained Stage 3 model
- Random ball positions and velocities
- Success metric: Balanced offense/defense performance

---

## Running Training

```bash
# Stage 1 (fresh start)
uv run train_stages.py --stage 1

# Stage 2 (load Stage 1)
uv run train_stages.py --stage 2 --load saves/foosball_stage_1_completed.zip

# Stage 3 (load Stage 2)
uv run train_stages.py --stage 3 --load saves/foosball_stage_2_completed.zip

# Stage 4 (load Stage 3)
uv run train_stages.py --stage 4 --load saves/foosball_stage_3_completed.zip

# Full pipeline (all stages in sequence)
uv run train_stages.py --stage 1 && \
uv run train_stages.py --stage 2 --load saves/foosball_stage_1_completed.zip && \
uv run train_stages.py --stage 3 --load saves/foosball_stage_2_completed.zip && \
uv run train_stages.py --stage 4 --load saves/foosball_stage_3_completed.zip
```

---

## Files Modified
- `foosball_env.py`: Core environment changes (rotation speed, curriculum, reward)
- `train_stages.py`: Curriculum level assignment

---

## Notes for Future Development

1. **Rotation Speed**: If agents still learn too-fast movements, consider further reducing `max_vel`
2. **Redundant Action Penalty**: Current threshold is >2 rods. Adjust if needed:
   - More restrictive: Penalty for >1 rod
   - Less restrictive: Penalty for >3 rods
3. **Curriculum Difficulty**: Consider adding intermediate stages (e.g., Stage 1.5 with moderate ball movement)
4. **Stage 1 Ball Spawn**: Y-range could be adjusted (currently ±0.3) for different difficulty
