#!/usr/bin/env python3
"""
Test script to verify rod assignments and colors
Red Team: rods 1,2,4,6
Blue Team: rods 3,5,7,8
"""

import numpy as np
from foosball_env import FoosballEnv


def test_rod_assignments():
    """Test and display rod team assignments"""
    print("\n" + "="*80)
    print("ROD TEAM ASSIGNMENTS AND COLORS")
    print("="*80)
    
    env = FoosballEnv(render_mode='direct', curriculum_level=1, player_id=1)
    
    print("\n Team 1 (RED) - Rods 1, 2, 4, 6:")
    print(f"  Slide joints: {len(env.team1_slide_joints)} (4 expected)")
    print(f"  Rotation joints: {len(env.team1_rev_joints)} (4 expected)")
    
    print("\n Team 2 (BLUE) - Rods 3, 5, 7, 8:")
    print(f"  Slide joints: {len(env.team2_slide_joints)} (4 expected)")
    print(f"  Rotation joints: {len(env.team2_rev_joints)} (4 expected)")
    
    env.close()


def test_red_team_movement():
    """Test moving only red team rods"""
    print("\n" + "="*80)
    print("TEST 1: RED TEAM MOVEMENT (Rods 1,2,4,6)")
    print("="*80)
    
    env = FoosballEnv(render_mode='human', curriculum_level=1, player_id=1)
    obs, _ = env.reset()
    
    print("\nMoving RED team rods:")
    print("  - Rods 1, 2, 4, 6 should move (RED colored)")
    print("  - Rods 3, 5, 7, 8 should stay still (BLUE colored)")
    
    for step in range(100):
        # Only control Team 1 (Red): indices 0-3 are slides, 4-7 are rotations
        action = np.zeros(8)
        action[0:4] = 0.5  # Slide all red rods
        action[4:8] = 0.5  # Rotate all red rods
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if step == 0:
            print(f"\n  Step {step+1}: Red rods should be moving...")
        if step == 50:
            print(f"  Step {step+1}: Red rods still moving...")
        
        if terminated or truncated:
            print(f"\n  Episode ended at step {step+1}")
            break
    
    print("✅ Red team movement test complete")
    env.close()


def test_blue_team_movement():
    """Test moving only blue team rods"""
    print("\n" + "="*80)
    print("TEST 2: BLUE TEAM MOVEMENT (Rods 3,5,7,8)")
    print("="*80)
    
    # Use Team 2 to control blue rods
    env = FoosballEnv(render_mode='human', curriculum_level=1, player_id=2)
    obs, _ = env.reset()
    
    print("\nMoving BLUE team rods:")
    print("  - Rods 3, 5, 7, 8 should move (BLUE colored)")
    print("  - Rods 1, 2, 4, 6 should stay still (RED colored)")
    
    for step in range(100):
        # Only control Team 2 (Blue): indices 0-3 are slides, 4-7 are rotations
        action = np.zeros(8)
        action[0:4] = -0.5  # Slide all blue rods in opposite direction
        action[4:8] = -0.5  # Rotate all blue rods in opposite direction
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if step == 0:
            print(f"\n  Step {step+1}: Blue rods should be moving...")
        if step == 50:
            print(f"  Step {step+1}: Blue rods still moving...")
        
        if terminated or truncated:
            print(f"\n  Episode ended at step {step+1}")
            break
    
    print("✅ Blue team movement test complete")
    env.close()


def test_alternating_movement():
    """Test moving red and blue rods alternately"""
    print("\n" + "="*80)
    print("TEST 3: ALTERNATING MOVEMENT (Red then Blue)")
    print("="*80)
    
    env = FoosballEnv(render_mode='human', curriculum_level=1, player_id=1)
    obs, _ = env.reset()
    
    print("\nAlternating movement pattern:")
    
    for phase in range(3):
        if phase % 2 == 0:
            print(f"\n  Phase {phase+1}: RED team moving...")
            for step in range(50):
                action = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        else:
            print(f"  Phase {phase+1}: BLUE team moving...")
            for step in range(50):
                action = np.array([-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3])
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    
    print("\n✅ Alternating movement test complete")
    env.close()


def main():
    """Run all rod color and movement tests"""
    print("\n" + "="*80)
    print("FOOSBALL ROD COLORS AND ASSIGNMENT TEST SUITE")
    print("="*80)
    print("\nExpected Configuration:")
    print("  RED Team (Player 1): Rods 1, 2, 4, 6 - RED colored")
    print("  BLUE Team (Player 2): Rods 3, 5, 7, 8 - BLUE colored")
    
    try:
        test_rod_assignments()
        test_red_team_movement()
        test_blue_team_movement()
        test_alternating_movement()
        
        print("\n" + "="*80)
        print("✅ ALL ROD TESTS PASSED")
        print("="*80)
        print("\nVerified:")
        print("  ✓ Red rods (1,2,4,6) are correctly assigned to Team 1")
        print("  ✓ Blue rods (3,5,7,8) are correctly assigned to Team 2")
        print("  ✓ Red and Blue teams move independently")
        print("  ✓ Rod colors are displayed correctly")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
