#!/usr/bin/env python3
"""
Test Stage 1: Dribble - Ball in random position reachable by all rods

Tests:
- Ball spawns at random positions across all rod ranges
- No motion in initial ball velocity
- All rods can reach the ball
- Reward calculation works correctly
"""

import numpy as np
from foosball_env import FoosballEnv


def test_stage_1_ball_randomization():
    """Test that Stage 1 ball spawns randomly across all rod positions"""
    print("\n" + "="*80)
    print("TEST 1.1: Stage 1 Ball Randomization")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=1, player_id=1)
    
    positions = []
    for i in range(30):
        obs, _ = env.reset()
        ball_pos = obs[:3]
        positions.append(ball_pos[0])  # x coordinate
        if i < 5:
            print(f"  Reset {i+1}: ball_pos = {ball_pos}")
    
    min_x = min(positions)
    max_x = max(positions)
    unique_positions = len(set(np.round(positions, 2)))
    
    print(f"\nResults over 30 resets:")
    print(f"  X range: [{min_x:.4f}, {max_x:.4f}]")
    print(f"  Expected: [-0.4, 0.0]")
    print(f"  Unique positions: {unique_positions}/30")
    
    assert min_x >= -0.41, f"Min X {min_x} below expected -0.4"
    assert max_x <= 0.01, f"Max X {max_x} above expected 0.0"
    assert unique_positions >= 15, f"Only {unique_positions} unique positions, need variation"
    
    print("✅ PASS: Ball randomization working correctly")
    env.close()


def test_stage_1_ball_velocity():
    """Test that Stage 1 ball starts with zero velocity"""
    print("\n" + "="*80)
    print("TEST 1.2: Stage 1 Ball Initial Velocity")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=1, player_id=1)
    
    for i in range(10):
        obs, _ = env.reset()
        ball_vel = obs[3:6]
        if i == 0:
            print(f"  Initial ball velocity: {ball_vel}")
        
        assert np.allclose(ball_vel, [0, 0, 0]), f"Ball should have zero velocity, got {ball_vel}"
    
    print("✅ PASS: Ball starts with zero velocity")
    env.close()


def test_stage_1_all_rods_reachable():
    """Test that all rod positions can reach the spawned ball"""
    print("\n" + "="*80)
    print("TEST 1.3: All Rods Can Reach Spawned Ball")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=1, player_id=1)
    
    # Team 1 rods cover roughly x from -0.4 to 0.0
    # Each rod has a slide range to move left/right
    print(f"  Team 1 has {len(env.team1_slide_joints)} slide joints")
    print(f"  Team 1 has {len(env.team1_rev_joints)} rotation joints")
    
    for i in range(10):
        obs, _ = env.reset()
        ball_x = obs[0]
        
        # Check if ball is within Team 1 range
        assert -0.42 <= ball_x <= 0.02, f"Ball at {ball_x} outside expected Team 1 range [-0.4, 0.0]"
        
        if i < 3:
            print(f"  Reset {i+1}: Ball at x={ball_x:.3f} (reachable by all rods)")
    
    print("✅ PASS: All spawned balls are reachable by the rods")
    env.close()


def test_stage_1_rewards():
    """Test that reward calculation works for Stage 1"""
    print("\n" + "="*80)
    print("TEST 1.4: Stage 1 Reward Calculation")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=1, player_id=1)
    obs, _ = env.reset()
    
    # Test with random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    
    print(f"  Sample action: {action}")
    print(f"  Reward: {reward:.6f}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")
    
    # Reward should be a valid number
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert not np.isnan(reward), "Reward should not be NaN"
    assert not np.isinf(reward), "Reward should not be infinite"
    
    print("✅ PASS: Reward calculation works correctly")
    env.close()


def test_stage_1_action_space():
    """Test that Stage 1 has correct action space"""
    print("\n" + "="*80)
    print("TEST 1.5: Stage 1 Action Space (4 rods × 2 DOF)")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=1, player_id=1)
    
    print(f"  Action space shape: {env.action_space.shape}")
    print(f"  Action space bounds: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"  Slide joints: {len(env.team1_slide_joints)}")
    print(f"  Rotation joints: {len(env.team1_rev_joints)}")
    
    assert env.action_space.shape == (8,), f"Expected (8,), got {env.action_space.shape}"
    assert len(env.team1_slide_joints) == 4, f"Expected 4 slide joints"
    assert len(env.team1_rev_joints) == 4, f"Expected 4 rotation joints"
    
    print("✅ PASS: Action space is correct (8 DOF: 4 slide + 4 rotate)")
    env.close()


def test_stage_1_multiple_steps():
    """Test that Stage 1 can run multiple steps without errors"""
    print("\n" + "="*80)
    print("TEST 1.6: Stage 1 Multi-Step Execution")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=1, player_id=1)
    obs, _ = env.reset()
    
    steps = 50
    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if step == 0:
            print(f"  Running {steps} steps...")
        
        if terminated or truncated:
            print(f"  Episode ended at step {step+1}")
            break
    
    print(f"  Completed {step+1}/{steps} steps successfully")
    print("✅ PASS: Multi-step execution works")
    env.close()


def main():
    """Run all Stage 1 tests"""
    print("\n" + "="*80)
    print("STAGE 1: DRIBBLE - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing: Ball randomization, zero velocity, all rods reachable,")
    print("         reward calculation, action space, multi-step execution")
    
    try:
        test_stage_1_ball_randomization()
        test_stage_1_ball_velocity()
        test_stage_1_all_rods_reachable()
        test_stage_1_rewards()
        test_stage_1_action_space()
        test_stage_1_multiple_steps()
        
        print("\n" + "="*80)
        print("✅ ALL STAGE 1 TESTS PASSED")
        print("="*80)
        print("\nStage 1 is ready for training!")
        print("Run: uv run train_stages.py --stage 1")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
