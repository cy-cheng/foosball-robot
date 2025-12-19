import time
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from foosball_env import FoosballEnv
from foosball_utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ws9.yaml")
    # Allow user to specify which update checkpoint to load (e.g., "update_10")
    parser.add_argument("--load_dir", type=str, required=True, help="Path to the directory containing agent checkpoints (e.g., saves/ws9/update_5)")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)

    print(f"🚀 Launching Test Session")
    print(f"   - Config: {args.config}")
    print(f"   - Checkpoint Dir: {args.load_dir}")

    # 1. Create Environment (Human Render Mode)
    # We use 'human' to visualize the game in PyBullet GUI
    env = FoosballEnv(config, render_mode='human', curriculum_level=1)
    
    # 2. Load the 4 Agents
    agent_roles = ["GK", "DEF", "MID", "STR"]
    agents = []
    
    print("🤖 Loading Agents...")
    for role in agent_roles:
        model_path = f"{args.load_dir}/{role}.zip"
        try:
            # We don't need to pass env here for inference only
            model = PPO.load(model_path)
            agents.append(model)
            print(f"   ✅ Loaded {role} from {model_path}")
        except FileNotFoundError:
            print(f"   ❌ Error: Could not find {role} agent at {model_path}")
            return

    # 3. Game Loop
    for episode in range(1, args.episodes + 1):
        print(f"\n=== Episode {episode} ===")
        obs, _ = env.reset() # Shape: (4, 7)
        
        done = False
        total_rewards = np.zeros(4)
        step_count = 0
        
        while not done:
            # A. Get Actions from all 4 Brains
            actions_list = []
            
            for i in range(4):
                # Slice observation for specific agent
                agent_obs = obs[i, :] 
                
                # Predict action (Deterministic for testing)
                action, _state = agents[i].predict(agent_obs, deterministic=True)
                actions_list.append(action)
            
            # Stack actions: (4, 2)
            env_actions = np.stack(actions_list)
            
            # B. Step Environment
            obs, rewards, term, trunc, info = env.step(env_actions)
            
            total_rewards += rewards
            step_count += 1
            
            if term or trunc:
                done = True
                
                # Basic Stats
                print(f"   Game Over at Step {step_count}")
                print(f"   Total Rewards: GK:{total_rewards[0]:.1f} | DEF:{total_rewards[1]:.1f} | MID:{total_rewards[2]:.1f} | STR:{total_rewards[3]:.1f}")
                
                if rewards[3] > 5.0: # Heuristic for scoring
                    print("   🎉 GOAL SCORED (Likely)!")
                elif rewards[0] < -5.0:
                    print("   💀 GOAL CONCEDED!")

    env.close()

if __name__ == "__main__":
    main()
