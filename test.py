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
    parser.add_argument("--load_dir", type=str, required=True, help="Path to checkpoint dir (e.g., saves/ws9/update_0)")
    parser.add_argument("--episodes", type=int, default=3)
    # Toggle this to see random movement on untrained agents
    parser.add_argument("--stochastic", action="store_true", help="Enable random exploration (use this for Update 0!)")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"🚀 Launching Test Session")
    print(f"   - Mode: {'STOCHASTIC (Random Noise)' if args.stochastic else 'DETERMINISTIC (Strict Policy)'}")
    
    env = FoosballEnv(config, render_mode='human', curriculum_level=1)
    
    agent_roles = ["GK", "DEF", "MID", "STR"]
    agents = []
    
    print("🤖 Loading Agents...")
    for role in agent_roles:
        try:
            model = PPO.load(f"{args.load_dir}/{role}.zip")
            agents.append(model)
            print(f"   ✅ Loaded {role}")
        except FileNotFoundError:
            print(f"   ❌ Failed to load {role}")
            return

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        
        # Identify the ONE active agent for this episode (Stage 1 Curriculum)
        active_idx = env.active_agent_idx
        active_role = agent_roles[active_idx]
        
        print(f"\n=== Episode {episode} (Active Agent: {active_role}) ===")
        
        while not done:
            actions_list = []
            
            # Get actions from all 4 brains
            for i in range(4):
                # deterministic=False means "Add Random Noise"
                # deterministic=True means "Output Mean (Usually 0.0 for untrained)"
                use_deterministic = not args.stochastic
                action, _ = agents[i].predict(obs[i], deterministic=use_deterministic)
                actions_list.append(action)
            
            env_actions = np.stack(actions_list)
            
            # --- DEBUG: Print what the Active Agent WANTS to do ---
            # If this prints numbers but rod doesn't move -> Physics Broken.
            # If this prints 0.00 -> Agent is boring.
            if step % 10 == 0: # Print every 10 steps to reduce spam
                act = env_actions[active_idx]
                print(f"   Step {step}: {active_role} Output -> Slide: {act[0]:.3f} | Rot: {act[1]:.3f}")

            obs, rewards, term, trunc, info = env.step(env_actions)
            step += 1
            
            if term or trunc:
                done = True
                print(f"   🏁 Episode Finished. Total Steps: {step}")
                if info.get('goal_scored', 0) == 1:
                    print("   🎉 GOAL SCORED!")

    env.close()

if __name__ == "__main__":
    main()
