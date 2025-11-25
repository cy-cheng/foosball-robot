#!/usr/bin/env python3
"""
Symmetric two-agent evaluation script.
Loads a trained policy and has both teams play against each other.
"""

import numpy as np
import argparse
from stable_baselines3 import PPO
from foosball_env import FoosballEnv


class SymmetricMatch:
    """Run a symmetric foosball match between two identical agents"""
    
    def __init__(self, model_path, render=True):
        self.model = PPO.load(model_path)
        self.render = render
        self.env = FoosballEnv(
            render_mode='human' if render else 'human',
            curriculum_level=4,
            debug_mode=False,
            player_id=1
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
            team1_goals = 0
            team2_goals = 0
            
            while not done:
                # Team 1 action
                action1, _ = self.model.predict(obs, deterministic=True)
                
                # Mirror observation for Team 2
                mirrored_obs = obs.copy()
                mirrored_obs[0] = -obs[0]  # Mirror ball x
                mirrored_obs[3] = -obs[3]  # Mirror ball vel x
                
                # Team 2 action (mirror of team 1)
                action2, _ = self.model.predict(mirrored_obs, deterministic=True)
                # Don't need to mirror action back since env does it
                
                obs, reward, terminated, truncated, _ = self.env.step(action1, opponent_action=action2)
                
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
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
    parser = argparse.ArgumentParser(description="Test symmetric foosball agents")
    parser.add_argument("--model", default="saves/foosball_model_final.zip", help="Model path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
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
