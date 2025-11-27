#!/usr/bin/env python3
"""
Symmetric two-agent evaluation script.
Loads a trained policy and has both teams play against each other.
"""

import numpy as np
import argparse
import time
from stable_baselines3 import PPO
from foosball_env import FoosballEnv


def test_individual_rod_control():
    """Test moving each rod of a team individually."""
    print("\n" + "="*80)
    print("TEST: INDIVIDUAL ROD CONTROL (Team 1 - RED)")
    print("="*80)
    
    env = FoosballEnv(render_mode='human', curriculum_level=1, player_id=1)
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


class SymmetricMatch:
    """Run a symmetric foosball match between two identical agents"""
    
    def __init__(self, model_path, render=True):
        self.model = PPO.load(model_path)
        self.render = render
        self.env = FoosballEnv(
            render_mode='human' if render else 'computer',
            curriculum_level=4,
            debug_mode=False
        )
    
    def run_match(self, num_episodes=5, verbose=True):
        """Run match with statistics"""
        stats = {
            'team1_goals': [],
            'team2_goals': [],
            'episode_lengths': [],
            'episode_rewards': []
        }
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Team 1 action
                action1, _ = self.model.predict(obs, deterministic=True)
                
                # Mirror observation for Team 2
                mirrored_obs = self.env.get_mirrored_obs()
                
                # Team 2 action (mirror of team 1)
                action2, _ = self.model.predict(mirrored_obs, deterministic=True)
                
                obs, reward, terminated, truncated, _ = self.env.step(action1, opponent_action=action2)
                
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            # This part is tricky because the env is from perspective of player 1
            # I will assume goals_this_level is for player 1
            team1_goals = self.env.goals_this_level
            stats['team1_goals'].append(team1_goals)
            stats['episode_lengths'].append(episode_length)
            stats['episode_rewards'].append(episode_reward)
            
            if verbose:
                print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}, Goals={team1_goals}")
        
        return stats
    
    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Test foosball environment")
    parser.add_argument("--model", default="saves/foosball_stage_4_completed.zip", help="Model path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-rods", action="store_true", help="Test individual rod control")
    
    args = parser.parse_args()
    
    if args.test_rods:
        test_individual_rod_control()
        return

    np.random.seed(args.seed)
    
    print(f"🎮 Foosball Symmetric Agent Match")
    print(f"   Model: {args.model}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Render: {not args.no_render}")
    print()
    
    match = SymmetricMatch(args.model, render=not args.no_render)
    stats = match.run_match(num_episodes=args.episodes, verbose=True)
    match.close()
    
    print(f"\n📊 Match Statistics:")
    print(f"   Avg Goals: {np.mean(stats['team1_goals']):.2f} ± {np.std(stats['team1_goals']):.2f}")
    print(f"   Avg Episode Length: {np.mean(stats['episode_lengths']):.1f}")
    print(f"   Avg Reward: {np.mean(stats['episode_rewards']):.2f} ± {np.std(stats['episode_rewards']):.2f}")
    print(f"   Total Goals: {sum(stats['team1_goals'])}")


if __name__ == "__main__":
    main()
