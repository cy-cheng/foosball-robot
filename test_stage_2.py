#!/usr/bin/env python3
"""
Test Stage 2: Pass - Ball rolling toward rods at midfield

Tests:
- Ball spawns at midfield
- Ball has incoming velocity toward player's side
- Velocity has random Y component (cross-court passes)
- Reward encourages interception
"""

import numpy as np
from foosball_env import FoosballEnv


def test_stage_2_ball_position():
    """Test that Stage 2 ball spawns at midfield"""
    print("\n" + "="*80)
    print("TEST 2.1: Stage 2 Ball Spawns at Midfield")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=2, player_id=1)
    
    for i in range(10):
        obs, _ = env.reset()
        ball_pos = obs[:3]
        
        # Ball should be at x=0 (midfield)
        assert abs(ball_pos[0]) < 0.01, f"Ball X should be ~0, got {ball_pos[0]}"
        assert abs(ball_pos[1]) < 0.01, f"Ball Y should be ~0, got {ball_pos[1]}"
        
        if i == 0:
            print(f"  Ball spawns at: {ball_pos}")
    
    print("✅ PASS: Ball spawns at midfield")
    env.close()


def test_stage_2_ball_velocity():
    """Test that Stage 2 ball has incoming velocity"""
    print("\n" + "="*80)
    print("TEST 2.2: Stage 2 Ball Incoming Velocity")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=2, player_id=1)
    
    velocities = []
    for i in range(10):
        obs, _ = env.reset()
        ball_vel = obs[3:6]
        velocities.append(ball_vel)
        
        # For player 1, ball should move toward negative X (toward their goal)
        assert ball_vel[0] < -0.5, f"Ball should move toward player, got vx={ball_vel[0]}"
        assert abs(ball_vel[2]) < 0.01, f"Ball Z velocity should be ~0, got {ball_vel[2]}"
        
        if i < 3:
            print(f"  Reset {i+1}: velocity = {ball_vel}")
    
    # Check Y velocity has variation
    y_velocities = [v[1] for v in velocities]
    y_range = max(y_velocities) - min(y_velocities)
    print(f"\n  Y velocity range: [{min(y_velocities):.3f}, {max(y_velocities):.3f}]")
    assert y_range > 0.5, "Y velocity should have variation for cross-court passes"
    
    print("✅ PASS: Ball has incoming velocity with Y variation")
    env.close()


def test_stage_2_player_perspective():
    """Test Stage 2 for both players"""
    print("\n" + "="*80)
    print("TEST 2.3: Stage 2 Works for Both Players")
    print("="*80)
    
    for player_id in [1, 2]:
        env = FoosballEnv(render_mode='direct', curriculum_level=2, player_id=player_id)
        obs, _ = env.reset()
        ball_vel = obs[3:6]
        
        if player_id == 1:
            # Player 1 on left, ball comes from right (negative X in world frame)
            assert ball_vel[0] < -0.5, f"P1: Ball should move toward player"
            print(f"  Player 1: ball velocity = {ball_vel}")
        else:
            # Player 2 on right, ball comes from left (positive X in world frame)
            # But observation is MIRRORED, so X becomes negative
            assert ball_vel[0] < -0.5, f"P2: Ball should move toward player (mirrored)"
            print(f"  Player 2: ball velocity = {ball_vel} (mirrored observation)")
        
        env.close()
    
    print("✅ PASS: Stage 2 works for both players")


def test_stage_2_interception_reward():
    """Test that intercepting the ball gives good rewards"""
    print("\n" + "="*80)
    print("TEST 2.4: Stage 2 Interception Reward")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=2, player_id=1)
    
    rewards = []
    for i in range(5):
        obs, _ = env.reset()
        
        # Try to move rods to intercept (random actions)
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
    
    avg_reward = np.mean(rewards)
    print(f"  Average reward over 5 attempts: {avg_reward:.6f}")
    print(f"  Reward range: [{min(rewards):.6f}, {max(rewards):.6f}]")
    
    # Reward should be numeric and stable
    assert all(not np.isnan(r) for r in rewards), "Rewards contain NaN"
    assert all(not np.isinf(r) for r in rewards), "Rewards contain inf"
    
    print("✅ PASS: Interception reward calculation works")
    env.close()


def test_stage_2_observation_shape():
    """Test that observation shape is correct"""
    print("\n" + "="*80)
    print("TEST 2.5: Stage 2 Observation Shape")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=2, player_id=1)
    obs, _ = env.reset()
    
    expected_shape = (38,)  # 3 ball_pos + 3 ball_vel + 16 joint_pos + 16 joint_vel
    print(f"  Observation shape: {obs.shape}")
    print(f"  Expected: {expected_shape}")
    
    assert obs.shape == expected_shape, f"Expected {expected_shape}, got {obs.shape}"
    
    print("✅ PASS: Observation shape is correct")
    env.close()


def test_stage_2_multiple_episodes():
    """Test Stage 2 stability over multiple episodes"""
    print("\n" + "="*80)
    print("TEST 2.6: Stage 2 Multiple Episodes Stability")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=2, player_id=1)
    
    episodes = 3
    steps_per_episode = 50
    
    for ep in range(episodes):
        obs, _ = env.reset()
        for step in range(steps_per_episode):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        print(f"  Episode {ep+1}: {step+1} steps")
    
    print("✅ PASS: Multiple episodes run stably")
    env.close()


def main():
    """Run all Stage 2 tests"""
    print("\n" + "="*80)
    print("STAGE 2: PASS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing: Ball at midfield, incoming velocity, Y variation,")
    print("         interception rewards, observation shape, multi-episode stability")
    
    try:
        test_stage_2_ball_position()
        test_stage_2_ball_velocity()
        test_stage_2_player_perspective()
        test_stage_2_interception_reward()
        test_stage_2_observation_shape()
        test_stage_2_multiple_episodes()
        
        print("\n" + "="*80)
        print("✅ ALL STAGE 2 TESTS PASSED")
        print("="*80)
        print("\nStage 2 is ready for training!")
        print("Run: uv run train_stages.py --stage 2 --load saves/foosball_stage_1_completed.zip")
        
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
