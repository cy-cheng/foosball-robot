# Foosball RL Training - Test Suite Documentation

## Overview

Comprehensive test suites for all 4 curriculum training stages. Each stage has its own test file with multiple validation checks.

## Test Files

### Individual Stage Tests

- **test_stage_1.py** - Dribble stage tests
- **test_stage_2.py** - Pass stage tests  
- **test_stage_3.py** - Defend stage tests
- **test_stage_4.py** - Full game stage tests

### Master Test Runner

- **test_all_stages.py** - Runs all 4 stage tests and provides summary

## Running Tests

### Run All Tests
```bash
python3 test_all_stages.py
```

### Run Individual Stage Tests
```bash
python3 test_stage_1.py
python3 test_stage_2.py
python3 test_stage_3.py
python3 test_stage_4.py
```

## Stage 1: Dribble Tests

**Purpose**: Verify Stage 1 curriculum trains all rods equally

### Tests Included
1. **Ball Randomization** - Confirms ball spawns randomly across all rod positions [-0.4, 0.0]
2. **Ball Initial Velocity** - Verifies ball starts with zero velocity
3. **All Rods Reachable** - Ensures spawned balls are within reach of all rods
4. **Reward Calculation** - Validates reward function works correctly
5. **Action Space** - Confirms 8-DOF action space (4 rods × 2 DOF)
6. **Multi-Step Execution** - Tests environment runs for multiple steps

### Expected Results
```
✅ Ball randomization working correctly
✅ Ball starts with zero velocity
✅ All spawned balls are reachable by the rods
✅ Reward calculation works correctly
✅ Action space is correct (8 DOF: 4 slide + 4 rotate)
✅ Multi-step execution works
```

---

## Stage 2: Pass Tests

**Purpose**: Verify Stage 2 curriculum trains interception

### Tests Included
1. **Ball at Midfield** - Confirms ball spawns at x=0, y=0
2. **Incoming Velocity** - Verifies ball moves toward player with speed=-1.0
3. **Y Variation** - Ensures Y velocity varies for cross-court passes
4. **Player Symmetry** - Confirms works for both Player 1 and Player 2
5. **Interception Reward** - Validates reward for interception attempts
6. **Observation Shape** - Checks observation is correct shape (38,)
7. **Multi-Episode Stability** - Tests multiple episodes run stably

### Expected Results
```
✅ Ball spawns at midfield
✅ Ball has incoming velocity with Y variation
✅ Stage 2 works for both players
✅ Interception reward calculation works
✅ Observation shape is correct
✅ Multiple episodes run stably
```

---

## Stage 3: Defend Tests

**Purpose**: Verify Stage 3 curriculum trains defensive positioning

### Tests Included
1. **Opponent-Side Spawn** - Confirms ball spawns on opponent's side (x=0.4)
2. **Fast Velocity** - Verifies ball has high speed (vel_x=-5.0)
3. **Difficulty Progression** - Confirms Stage 3 is harder than Stage 2
4. **Defensive Challenge** - Validates defensive scenario functionality
5. **Player Symmetry** - Ensures works for both players with mirrored observations
6. **Multi-Episode Stability** - Tests stability across multiple episodes

### Expected Results
```
✅ Ball spawns on opponent's side
✅ Ball has fast velocity
✅ Stage 3 is harder than Stage 2
✅ Stage 3 defensive challenge scenario works
✅ Stage 3 works for both players with symmetry
✅ Stage 3 runs stably
```

---

## Stage 4: Full Game Tests

**Purpose**: Verify Stage 4 curriculum trains complete game play

### Tests Included
1. **Ball Randomization** - Confirms ball spawns anywhere on table
2. **Velocity Randomization** - Verifies velocity is random in any direction
3. **Difficulty Progression** - Confirms Stage 4 is hardest stage
4. **Full Game Dynamics** - Validates mix of offensive and defensive scenarios
5. **Episode Completion** - Tests full episodes can complete properly
6. **Player Symmetry** - Ensures works for both players
7. **Validity Checks** - Confirms observations and rewards are valid throughout

### Expected Results
```
✅ Ball spawns randomly across table
✅ Ball has random velocity in any direction
✅ Stage 4 is harder than Stage 3
✅ Stage 4 covers offensive and defensive scenarios
✅ Full game episodes execute properly
✅ Stage 4 works for both players
✅ Observations and rewards valid throughout
```

---

## Test Coverage

### What Tests Verify

✅ **Curriculum Correctness**
- Ball spawning matches stage expectations
- Velocity conditions match stage difficulty
- Progressive difficulty from Stage 1 → 4

✅ **Environment Stability**
- Episodes can run for multiple steps
- Observations remain valid
- Rewards are calculated correctly
- No NaN or Inf values

✅ **Symmetry**
- Both Player 1 and Player 2 work correctly
- Observations are properly mirrored for Player 2
- Curriculum applies correctly to both players

✅ **Action Space**
- 8-dimensional action space (4 rods × 2 DOF)
- Actions are properly scaled
- Motor control works correctly

---

## Test Results

When all tests pass, you'll see:

```
================================================================================
✅ ALL STAGES PASSED - SYSTEM READY FOR TRAINING
================================================================================

Next steps:
  1. uv run train_stages.py --stage 1
  2. uv run train_stages.py --stage 2 --load saves/foosball_stage_1_completed.zip
  3. uv run train_stages.py --stage 3 --load saves/foosball_stage_2_completed.zip
  4. uv run train_stages.py --stage 4 --load saves/foosball_stage_3_completed.zip
```

---

## Troubleshooting

### Test Fails with "Ball not reachable"
- Check that curriculum ball spawn ranges match rod positions
- Verify rod joint limits in foosball.urdf

### Test Fails with "NaN in reward"
- Check reward calculation doesn't divide by zero
- Verify all joint states are valid

### Test Fails with "Shape mismatch"
- Confirm observation space is 38D (3 + 3 + 16 + 16)
- Check observation concatenation in _get_obs()

### Tests Fail Intermittently
- Some tests are probabilistic (e.g., goal detection)
- Run multiple times to see if it's consistent
- Check system resources and pybullet stability

---

## Continuous Integration

To integrate into CI/CD pipeline:

```bash
#!/bin/bash
cd foosball_robot
python3 test_all_stages.py
exit $?
```

Success when exit code is 0, failure when non-zero.

---

## Performance Notes

- Full test suite takes ~4-5 minutes per run
- Each stage test takes ~1 minute
- Stage 1 is fastest (simple spawning)
- Stage 4 is slowest (full randomization)
- All tests use headless mode (no GUI) for speed

---

## Test Improvements

Consider adding in future:

1. **Performance tests** - Measure simulation speed
2. **Visualization tests** - Screenshot comparisons
3. **Stress tests** - Run 1000s of episodes
4. **Memory tests** - Track memory usage
5. **Benchmark comparisons** - Compare with baseline metrics
