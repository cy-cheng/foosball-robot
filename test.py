#!/usr/bin/env python3
"""
Symmetric two-agent evaluation script.
Loads a trained policy and has both teams play against each other.
"""

import numpy as np
import argparse
import time
import pybullet as p
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from foosball_env import FoosballEnv
from foosball_utils import load_config
from train import DEFAULT_CONFIG_PATH


def test_individual_rod_control(config):
    """Test moving each rod of a team individually."""
    print("\n" + "="*80)
    print("TEST: INDIVIDUAL ROD CONTROL (Team 1 - RED)")
    print("="*80)
    
    env = FoosballEnv(config=config, render_mode='human', curriculum_level=1, player_id=1)
    obs, _ = env.reset()
    
    # Action indices for Team 1:
    # slide: 0, 1, 2, 3
    # rotate: 4, 5, 6, 7
    
    num_rods = 4
    
    for i in range(num_rods):
        # Test rotation for each rod
        
        # Clockwise rotation
        print(f"\nRod {i+1} (rotate index {i+4}): Spinning CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = 1.0  # Max rotation
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
            
        # Counter-clockwise rotation
        print(f"Rod {i+1} (rotate index {i+4}): Spinning COUNTER-CLOCKWISE")
        action = np.zeros(8)
        action[i+4] = -1.0 # Min rotation
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
            
        # Stop rotation
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)
        
    for i in range(num_rods):
        # Test sliding for each rod
        
        # Slide in
        print(f"\nRod {i+1} (slide index {i}): Sliding IN")
        action = np.zeros(8)
        action[i] = 1.0 # Max slide
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)
            
        # Slide out
        print(f"Rod {i+1} (slide index {i}): Sliding OUT")
        action = np.zeros(8)
        action[i] = -1.0 # Min slide
        for _ in range(50):
            env.step(action)
            time.sleep(0.01)

        # Center
        action = np.zeros(8)
        env.step(action)
        time.sleep(0.2)

    print("\n✅ Individual rod control test complete")
    env.close()


class StageMatch:
    """Run a match for a specific curriculum stage."""

    def __init__(self, model_path, config, stage=4, render=True):
        self.render = render
        self.stage = stage

        # Create the raw environment
        self.raw_env = FoosballEnv(
            config=config,
            render_mode='human' if render else 'direct',
            curriculum_level=self.stage,
            debug_mode=False,
        )

        # Wrap it in a DummyVecEnv
        self.env = DummyVecEnv([lambda: self.raw_env])

        # Load normalization stats if they exist
        vec_normalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vec_normalize_path):
            print(f"✅ Loading VecNormalize stats from: {vec_normalize_path}")
            self.env = VecNormalize.load(vec_normalize_path, self.env)
            # Set training to False to avoid updating the stats
            self.env.training = False
            # Don't normalize rewards
            self.env.norm_reward = False
        else:
            print("⚠️  Warning: VecNormalize stats not found. Running with unnormalized observations.")

        self.model = PPO.load(model_path, env=self.env)

    def run_match(self, num_episodes=5, verbose=True):
        """Run match with statistics"""
        stats = {
            'goals': [],
            'own_goals': [],
            'episode_lengths': [],
            'episode_rewards': []
        }

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            goals = 0
            own_goals = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)

                opponent_action = None
                if self.stage == 4 and isinstance(self.env, VecNormalize):
                    unnormalized_obs = self.env.unnormalize_obs(obs)
                    mirrored_unnormalized_obs = self.raw_env.get_mirrored_obs_from_unnormalized_obs(unnormalized_obs[0])
                    mirrored_obs = self.env.normalize_obs(np.expand_dims(mirrored_unnormalized_obs, axis=0))
                    opponent_action, _ = self.model.predict(mirrored_obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.raw_env.step(action[0], opponent_action=opponent_action[0])
                else:
                    if self.stage == 4:
                        print("⚠️ Warning: Cannot run self-play without VecNormalize stats. Opponent will be the simple bot.")
                    obs, reward, terminated, truncated, info = self.raw_env.step(action[0])

                if isinstance(self.env, VecNormalize):
                    obs = self.env.normalize_obs(np.expand_dims(obs, axis=0))
                else:
                    obs = np.expand_dims(obs, axis=0)

                episode_reward += reward
                episode_length += 1

                if episode_length % 1000 == 0 and verbose:
                    print(f"  Step {episode_length}: Current Reward={episode_reward:.2f}")

                done = terminated or truncated

            if done:
                ball_pos, _ = p.getBasePositionAndOrientation(self.raw_env.ball_id, physicsClientId=self.raw_env.client)
                if self.raw_env.player_id == 1:
                    if ball_pos[0] > self.raw_env.goal_line_x_2:
                        goals += 1
                    elif ball_pos[0] < self.raw_env.goal_line_x_1:
                        own_goals += 1
                else: # player_id == 2
                    if ball_pos[0] < self.raw_env.goal_line_x_1:
                        goals += 1
                    elif ball_pos[0] > self.raw_env.goal_line_x_2:
                        own_goals += 1

            stats['goals'].append(goals)
            stats['own_goals'].append(own_goals)
            stats['episode_lengths'].append(episode_length)
            stats['episode_rewards'].append(episode_reward)

            if verbose:
                print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}, Goals={goals}, Own Goals={own_goals}")

        return stats

    def close(self):
        self.env.close()



def test_torque_and_gravity(config):
    """Test the torque control and the effect of gravity on the rods."""
    print("\n" + "="*80)
    print("TEST: TORQUE CONTROL AND GRAVITY")
    print("="*80)
    
    env = FoosballEnv(config=config, render_mode='human', curriculum_level=1, player_id=1)
    obs, _ = env.reset()
    
    # Choose a rod to test (e.g., the first midfield rod, index 2)
    rod_action_index = 4 + 2  # Rotation action for the 3rd rod
    
    # Apply a positive torque to make it spin
    print("\nApplying positive torque...")
    action = np.zeros(8)
    action[rod_action_index] = 1.0  # Max torque
    for _ in range(100):
        env.step(action)
        time.sleep(0.01)
        
    # Apply zero torque and watch it settle
    print("\nApplying zero torque. Rod should settle to the bottom due to gravity.")
    action = np.zeros(8)
    for _ in range(300):
        env.step(action)
        time.sleep(0.01)
        
    # Apply a negative torque
    print("\nApplying negative torque...")
    action = np.zeros(8)
    action[rod_action_index] = -1.0  # Min torque
    for _ in range(100):
        env.step(action)
        time.sleep(0.01)
        
    # Apply zero torque and watch it settle again
    print("\nApplying zero torque. Rod should settle to the bottom.")
    action = np.zeros(8)
    for _ in range(300):
        env.step(action)
        time.sleep(0.01)

    print("\n✅ Torque and gravity test complete.")
    env.close()


def debug_single_rod_torque(config):
    """
    Apply a constant torque to a single rod and print out detailed physics info.
    """
    print("\n" + "="*80)
    print("DEBUG: SINGLE ROD TORQUE AND PHYSICS")
    print("="*80)
    
    env = FoosballEnv(config=config, render_mode='human', curriculum_level=1, player_id=1)
    env.reset()

    # Isolate a single rod - the first midfield rod for Team 1
    rod_joint_index = env.team1_rev_joints[2] 
    
    # Player mass and CoM offset from URDF
    player_mass = 0.1  # kg
    player_com_offset = 0.07  # meters
    g = abs(env.physics_config['gravity'])
    damping_coeff = env.env_config['rod_angular_damping']
    
    # Reset the joint to horizontal (pi/2) to see max effect of gravity
    p.resetJointState(env.table_id, rod_joint_index, np.pi/2, 0, physicsClientId=env.client)
    print(f"Resetting rod to horizontal (pi/2). Gravity should now pull it down.")
    time.sleep(2)

    # --- Test 1: No applied torque ---
    print("\n--- Test 1: Applying ZERO torque for 100 steps ---")
    action = np.zeros(8)
    for i in range(100):
        env.step(action)
        joint_state = p.getJointState(env.table_id, rod_joint_index, physicsClientId=env.client)
        angle, velocity = joint_state[0], joint_state[1]
        
        # Calculations for debugging
        grav_torque = -player_mass * g * player_com_offset * np.sin(angle)
        damping_torque = -damping_coeff * velocity
        net_torque = grav_torque + damping_torque
        
        if i % 10 == 0:
            print(f"Step {i:03d}: Angle={angle:.3f}, Vel={velocity:.3f} | GravT={grav_torque:.3f}, DampT={damping_torque:.3f}, NetT={net_torque:.3f}")
        time.sleep(0.01)

    # --- Test 2: Apply constant torque ---
    applied_torque = config['environment']['max_torque']
    print(f"\n--- Test 2: Applying CONSTANT torque ({applied_torque:.2f}) for 150 steps ---")
    action = np.zeros(8)
    action[4+2] = 1.0 # Max torque on the chosen rod

    for i in range(150):
        # We need to get the state *before* the step to calculate expected torque
        joint_state = p.getJointState(env.table_id, rod_joint_index, physicsClientId=env.client)
        angle, velocity = joint_state[0], joint_state[1]
        
        grav_torque = -player_mass * g * player_com_offset * np.sin(angle)
        damping_torque = -damping_coeff * velocity
        net_torque = applied_torque + grav_torque + damping_torque
        
        # Apply the action
        env.step(action)
        
        # Get the new state
        new_joint_state = p.getJointState(env.table_id, rod_joint_index, physicsClientId=env.client)
        new_angle, new_velocity = new_joint_state[0], new_joint_state[1]

        if i % 10 == 0:
            print(f"Step {i:03d}: Angle={new_angle:.3f}, Vel={new_velocity:.3f} | AppliedT={applied_torque:.3f}, GravT={grav_torque:.3f}, DampT={damping_torque:.3f}, NetT={net_torque:.3f}")
        time.sleep(0.01)

    print("\n✅ Debug test complete. Review the angle and velocity values in the logs.")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Test foosball environment")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training configuration YAML file"
    )
    parser.add_argument("--model", default="saves/foosball_stage_4_completed.zip", help="Model path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-rods", action="store_true", help="Test individual rod control")
    parser.add_argument("--test-torque", action="store_true", help="Test torque control and gravity")
    parser.add_argument("--debug-torque", action="store_true", help="Run a detailed torque physics debug test.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="Curriculum stage to test (1-4)")
    
    args = parser.parse_args()

    full_config = load_config(args.config)
    
    if args.test_rods:
        test_individual_rod_control(full_config)
        return
        
    if args.test_torque:
        test_torque_and_gravity(full_config)
        return

    if args.debug_torque:
        debug_single_rod_torque(full_config)
        return

    np.random.seed(args.seed)
    
    stage_to_test = args.stage if args.stage else 4

    print(f"🎮 Foosball Agent Match - Stage {stage_to_test}")
    print(f"   Model: {args.model}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Render: {not args.no_render}")
    print()
    
    match = StageMatch(args.model, config=full_config, stage=stage_to_test, render=not args.no_render)
    stats = match.run_match(num_episodes=args.episodes, verbose=True)
    match.close()
    
    print(f"\n📊 Match Statistics (Stage {stage_to_test}):")
    print(f"   Avg Goals: {np.mean(stats['goals']):.2f} ± {np.std(stats['goals']):.2f}")
    print(f"   Avg Own Goals: {np.mean(stats['own_goals']):.2f} ± {np.std(stats['own_goals']):.2f}")
    print(f"   Avg Episode Length: {np.mean(stats['episode_lengths']):.1f}")
    print(f"   Avg Reward: {np.mean(stats['episode_rewards']):.2f} ± {np.std(stats['episode_rewards']):.2f}")
    print(f"   Total Goals: {sum(stats['goals'])}")
    print(f"   Total Own Goals: {sum(stats['own_goals'])}")


if __name__ == "__main__":
    main()
