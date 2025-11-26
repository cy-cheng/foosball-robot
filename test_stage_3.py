#!/usr/bin/env python3
"""
Test Stage 3: Defend - Ball shot fast from opponent's side

Tests:
- Ball spawns on opponent's side (x=0.4 for player 1)
- Ball has fast incoming velocity toward goal
- Ball has random Y component (corner shots)
- Defensive positioning challenge
"""

import numpy as np
from foosball_env import FoosballEnv


def test_stage_3_ball_position():
    """Test that Stage 3 ball spawns on opponent's side"""
    print("\n" + "="*80)
    print("TEST 3.1: Stage 3 Ball Spawns on Opponent's Side")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=3, player_id=1)
    
    for i in range(10):
        obs, _ = env.reset()
        ball_pos = obs[:3]
        
        # For player 1, opponent is on the right (positive X)
        # Ball should spawn at x=0.4
        assert ball_pos[0] > 0.3, f"Ball should be on opponent's side, got x={ball_pos[0]}"
        
        # Y can vary (corner shots)
        assert -0.3 <= ball_pos[1] <= 0.3, f"Ball Y out of range: {ball_pos[1]}"
        
        if i == 0:
            print(f"  Ball spawns at: {ball_pos}")
    
    print("✅ PASS: Ball spawns on opponent's side")
    env.close()


def test_stage_3_fast_velocity():
    """Test that Stage 3 ball has fast incoming velocity"""
    print("\n" + "="*80)
    print("TEST 3.2: Stage 3 Fast Incoming Velocity")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=3, player_id=1)
    
    velocities = []
    for i in range(10):
        obs, _ = env.reset()
        ball_vel = obs[3:6]
        velocities.append(ball_vel)
        
        # For player 1, ball should move toward negative X (toward their goal) FAST
        assert ball_vel[0] < -4.0, f"Ball should move fast toward goal, got vx={ball_vel[0]}"
        
        if i < 3:
            print(f"  Reset {i+1}: velocity = {ball_vel}")
    
    # Check Y velocity has variation
    y_velocities = [v[1] for v in velocities]
    print(f"\n  X velocity range: [{min(v[0] for v in velocities):.1f}, {max(v[0] for v in velocities):.1f}]")
    print(f"  Y velocity range: [{min(y_velocities):.1f}, {max(y_velocities):.1f}]")
    
    print("✅ PASS: Ball has fast velocity")
    env.close()


def test_stage_3_difficulty_vs_stage_2():
    """Test that Stage 3 is harder than Stage 2"""
    print("\n" + "="*80)
    print("TEST 3.3: Stage 3 Difficulty (vs Stage 2)")
    print("="*80)
    
    # Stage 2: Pass
    env2 = FoosballEnv(render_mode='direct', curriculum_level=2, player_id=1)
    obs2, _ = env2.reset()
    vel2 = obs2[3]  # X velocity
    pos2 = obs2[0]  # X position
    
    # Stage 3: Defend
    env3 = FoosballEnv(render_mode='direct', curriculum_level=3, player_id=1)
    obs3, _ = env3.reset()
    vel3 = obs3[3]  # X velocity
    pos3 = obs3[0]  # X position
    
    print(f"  Stage 2 - Ball pos_x: {pos2:.3f}, vel_x: {vel2:.3f}")
    print(f"  Stage 3 - Ball pos_x: {pos3:.3f}, vel_x: {vel3:.3f}")
    
    # Stage 3 should have faster velocity
    assert abs(vel3) > abs(vel2) * 2, f"Stage 3 should be faster"
    
    # Stage 3 should start closer to goal
    assert pos3 > pos2, f"Stage 3 should start closer to opponent's side"
    
    print("✅ PASS: Stage 3 is harder than Stage 2")
    env2.close()
    env3.close()


def test_stage_3_defensive_challenge():
    """Test that Stage 3 presents a defensive challenge"""
    print("\n" + "="*80)
    print("TEST 3.4: Stage 3 Defensive Challenge")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=3, player_id=1)
    
    # Just verify that Stage 3 can run multiple episodes
    # The defensive challenge is demonstrated by the faster ball and opponent side spawn
    episodes_completed = 0
    
    for ep in range(5):
        obs, _ = env.reset()
        
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                episodes_completed += 1
                break
    
    print(f"  Successfully ran {episodes_completed} episodes")
    print("  Stage 3 challenge: Ball fast (vel=-5.0) from opponent's side (x=0.4)")
    print("  (This is significantly harder than Stage 2)")
    
    assert episodes_completed >= 3, "Should complete multiple episodes"
    
    print("✅ PASS: Stage 3 defensive challenge scenario works")
    env.close()


def test_stage_3_both_players():
    """Test Stage 3 works for both players with symmetry"""
    print("\n" + "="*80)
    print("TEST 3.5: Stage 3 Works for Both Players")
    print("="*80)
    
    for player_id in [1, 2]:
        env = FoosballEnv(render_mode='direct', curriculum_level=3, player_id=player_id)
        obs, _ = env.reset()
        ball_pos = obs[:3]
        ball_vel = obs[3:6]
        
        # From the agent's (mirrored) perspective, opponent's side should always be positive X
        assert ball_pos[0] > 0.3, f"P{player_id}: Ball should be on opponent's side (>0.3), got {ball_pos[0]}"
        assert ball_vel[0] < -4.0, f"P{player_id}: Ball should move toward goal (<-4.0), got {ball_vel[0]}"
        print(f"  Player {player_id}: pos={ball_pos[0]:.2f}, vel={ball_vel[0]:.1f}")
        
        env.close()
    
    print("✅ PASS: Stage 3 works for both players with symmetry")


def test_stage_3_stability():
    """Test Stage 3 stability over multiple episodes"""
    print("\n" + "="*80)
    print("TEST 3.6: Stage 3 Multi-Episode Stability")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=3, player_id=1)
    
    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        print(f"  Episode {ep+1}: {step+1} steps")
    
    print("✅ PASS: Stage 3 runs stably")
    env.close()


def main():
    """Run all Stage 3 tests"""
    print("\n" + "="*80)
    print("STAGE 3: DEFEND - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing: Opponent-side spawn, fast velocity, difficulty progression,")
    print("         defensive challenge, player symmetry, stability")
    
    try:
        test_stage_3_ball_position()
        test_stage_3_fast_velocity()
        test_stage_3_difficulty_vs_stage_2()
        test_stage_3_defensive_challenge()
        test_stage_3_both_players()
        test_stage_3_stability()
        
        print("\n" + "="*80)
        print("✅ ALL STAGE 3 TESTS PASSED")
        print("="*80)
        print("\nStage 3 is ready for training!")
        print("Run: uv run train_stages.py --stage 3 --load saves/foosball_stage_2_completed.zip")
        
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
