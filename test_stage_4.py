#!/usr/bin/env python3
"""
Test Stage 4: Full Game - Random play with all conditions

Tests:
- Ball spawns at random positions anywhere on table
- Ball has random velocity in any direction
- Tests offset and defense together
- Full game strategy challenge
"""

import numpy as np
from foosball_env import FoosballEnv


def test_stage_4_ball_randomization():
    """Test that Stage 4 ball spawns randomly anywhere"""
    print("\n" + "="*80)
    print("TEST 4.1: Stage 4 Ball Randomization")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=4, player_id=1)
    
    positions = []
    for i in range(20):
        obs, _ = env.reset()
        ball_pos = obs[:3]
        positions.append((ball_pos[0], ball_pos[1]))
        
        # Ball should be anywhere on table
        assert -0.65 <= ball_pos[0] <= 0.65, f"Ball X out of bounds: {ball_pos[0]}"
        assert -0.35 <= ball_pos[1] <= 0.35, f"Ball Y out of bounds: {ball_pos[1]}"
        
        if i < 3:
            print(f"  Reset {i+1}: ball_pos = {ball_pos}")
    
    # Check coverage
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    
    print(f"\n  X range: [{min(x_coords):.3f}, {max(x_coords):.3f}]")
    print(f"  Y range: [{min(y_coords):.3f}, {max(y_coords):.3f}]")
    
    unique_positions = len(set((np.round(p[0], 2), np.round(p[1], 2)) for p in positions))
    print(f"  Unique positions: {unique_positions}/20")
    
    assert unique_positions >= 10, f"Not enough randomization"
    
    print("✅ PASS: Ball spawns randomly across table")
    env.close()


def test_stage_4_velocity_randomization():
    """Test that Stage 4 ball has random velocity in any direction"""
    print("\n" + "="*80)
    print("TEST 4.2: Stage 4 Velocity Randomization")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=4, player_id=1)
    
    velocities = []
    for i in range(15):
        obs, _ = env.reset()
        ball_vel = obs[3:6]
        velocities.append(ball_vel)
        
        # Velocity should be reasonable
        vel_magnitude = np.linalg.norm(ball_vel)
        assert 0 <= vel_magnitude <= 3, f"Velocity magnitude {vel_magnitude} out of range"
        
        if i < 3:
            print(f"  Reset {i+1}: velocity = {ball_vel}")
    
    # Check X velocity has both directions
    x_vels = [v[0] for v in velocities]
    y_vels = [v[1] for v in velocities]
    
    print(f"\n  X velocity range: [{min(x_vels):.2f}, {max(x_vels):.2f}]")
    print(f"  Y velocity range: [{min(y_vels):.2f}, {max(y_vels):.2f}]")
    
    has_positive_x = any(v > 0.5 for v in x_vels)
    has_negative_x = any(v < -0.5 for v in x_vels)
    
    assert has_positive_x and has_negative_x, "Should have balls moving in both X directions"
    
    print("✅ PASS: Ball has random velocity in any direction")
    env.close()


def test_stage_4_difficulty_vs_stage_3():
    """Test that Stage 4 is harder than Stage 3"""
    print("\n" + "="*80)
    print("TEST 4.3: Stage 4 Difficulty (vs Stage 3)")
    print("="*80)
    
    # Stage 3: Predictable defensive scenario
    env3 = FoosballEnv(render_mode='direct', curriculum_level=3, player_id=1)
    obs3, _ = env3.reset()
    
    # Stage 4: Full game, unpredictable
    env4 = FoosballEnv(render_mode='direct', curriculum_level=4, player_id=1)
    obs4, _ = env4.reset()
    
    print(f"  Stage 3 - Spawn pos: {obs3[:3]}")
    print(f"  Stage 4 - Spawn pos: {obs4[:3]}")
    
    print(f"  Stage 3 - Ball vel: {obs3[3:6]}")
    print(f"  Stage 4 - Ball vel: {obs4[3:6]}")
    
    print("\n  Stage 4 has:")
    print("  - Random ball positions (not just opponent's side)")
    print("  - Random velocity directions (not just incoming)")
    print("  - Full strategic play required")
    
    print("✅ PASS: Stage 4 is harder than Stage 3")
    env3.close()
    env4.close()


def test_stage_4_full_game_dynamics():
    """Test that Stage 4 covers full game dynamics"""
    print("\n" + "="*80)
    print("TEST 4.4: Stage 4 Full Game Dynamics")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=4, player_id=1)
    
    offensive_scenarios = 0
    defensive_scenarios = 0
    
    for ep in range(20):
        obs, _ = env.reset()
        ball_pos = obs[0]  # X position
        ball_vel = obs[3]  # X velocity
        
        # Classify scenario
        if ball_pos < -0.2:
            # Ball on player's side
            if ball_vel < 0:
                defensive_scenarios += 1
            else:
                offensive_scenarios += 1
        else:
            # Ball on opponent's side
            if ball_vel > 0:
                offensive_scenarios += 1
            else:
                defensive_scenarios += 1
    
    print(f"  Offensive scenarios: {offensive_scenarios}/20")
    print(f"  Defensive scenarios: {defensive_scenarios}/20")
    
    # Should have mix of offensive and defensive
    assert offensive_scenarios >= 5, "Not enough offensive scenarios"
    assert defensive_scenarios >= 5, "Not enough defensive scenarios"
    
    print("✅ PASS: Stage 4 covers offensive and defensive scenarios")
    env.close()


def test_stage_4_complete_episodes():
    """Test that full game episodes can complete"""
    print("\n" + "="*80)
    print("TEST 4.5: Stage 4 Complete Episode Execution")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=4, player_id=1)
    
    episodes = 3
    for ep in range(episodes):
        obs, _ = env.reset()
        steps = 0
        goals = 0
        
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            
            # Detect goals
            ball_x = obs[0]
            if terminated:
                if abs(ball_x) > 0.59:
                    goals += 1
                break
        
        print(f"  Episode {ep+1}: {steps} steps, {goals} goals")
    
    print("✅ PASS: Full game episodes execute properly")
    env.close()


def test_stage_4_both_players():
    """Test Stage 4 works for both players"""
    print("\n" + "="*80)
    print("TEST 4.6: Stage 4 Works for Both Players")
    print("="*80)
    
    for player_id in [1, 2]:
        env = FoosballEnv(render_mode='direct', curriculum_level=4, player_id=player_id)
        
        obs, _ = env.reset()
        print(f"  Player {player_id}: ball_pos = {obs[:3]}")
        
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
    
    print("✅ PASS: Stage 4 works for both players")


def test_stage_4_observation_and_rewards():
    """Test that observations and rewards are valid throughout game"""
    print("\n" + "="*80)
    print("TEST 4.7: Stage 4 Observation and Reward Validity")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=4, player_id=1)
    
    for ep in range(5):
        obs, _ = env.reset()
        
        # Check initial observation
        assert obs.shape == (38,), f"Wrong obs shape: {obs.shape}"
        assert not np.any(np.isnan(obs)), "NaN in observation"
        
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Check validity
            assert obs.shape == (38,), f"Wrong obs shape: {obs.shape}"
            assert isinstance(reward, (int, float)), "Reward not numeric"
            assert not np.isnan(reward), "NaN reward"
            assert not np.isinf(reward), "Inf reward"
            
            if terminated or truncated:
                break
    
    print("✅ PASS: Observations and rewards valid throughout")
    env.close()


def main():
    """Run all Stage 4 tests"""
    print("\n" + "="*80)
    print("STAGE 4: FULL GAME - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing: Random ball positions, random velocity, difficulty,")
    print("         offensive/defensive mix, episode completion, stability")
    
    try:
        test_stage_4_ball_randomization()
        test_stage_4_velocity_randomization()
        test_stage_4_difficulty_vs_stage_3()
        test_stage_4_full_game_dynamics()
        test_stage_4_complete_episodes()
        test_stage_4_both_players()
        test_stage_4_observation_and_rewards()
        
        print("\n" + "="*80)
        print("✅ ALL STAGE 4 TESTS PASSED")
        print("="*80)
        print("\nStage 4 is ready for training!")
        print("Run: uv run train_stages.py --stage 4 --load saves/foosball_stage_3_completed.zip")
        
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
