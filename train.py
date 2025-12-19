import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from foosball_env import FoosballEnv
from foosball_utils import load_config

# --- HELPER CLASSES ---

def make_env(rank, config):
    """Factory for SubprocVecEnv"""
    def _init():
        # Render mode 'direct' for speed on CPU workers
        return FoosballEnv(config, render_mode='direct', curriculum_level=1)
    return _init

# --- MAIN TRAINING LOOP ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ws9.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    # --- CONFIGURATION ---
    # Goal: Simulate 2048 Environments using 8 Processes (Accumulation)
    VIRTUAL_ENVS = 2048
    REAL_ENVS = 8  # Fixed to 8 as requested
    
    # Diversity horizon (how much unique data we want per update per virtual env)
    STEPS_PER_VIRTUAL_ENV = 512 
    
    # Calculate accumulation steps required per real process
    # Total Samples Target = 2048 * 512 = 1,048,576
    # n_steps per Real Env = 1,048,576 / 8 = 131,072 steps
    # This means the collection phase will be long, but the update will be massive.
    n_steps = int(STEPS_PER_VIRTUAL_ENV * (VIRTUAL_ENVS / REAL_ENVS))
    
    total_samples_per_update = n_steps * REAL_ENVS
    total_timesteps = 100_000_000
    
    print(f"🚀 Launching Massive-Batch Training")
    print(f"   - Virtual Envs: {VIRTUAL_ENVS}")
    print(f"   - Real Processes: {REAL_ENVS}")
    print(f"   - Collection Steps per Cycle: {n_steps}")
    print(f"   - Total Buffer Size: {total_samples_per_update:,} samples")

    # 1. Create Real Environment (Physics)
    # We pass this to PPO so it can infer shapes automatically (Fixes AttributeError)
    env = SubprocVecEnv([make_env(i, config) for i in range(REAL_ENVS)])
    
    # 2. Initialize 4 Agents (GK, DEF, MID, STR)
    agent_roles = ["GK", "DEF", "MID", "STR"]
    agents = []
    
    print("🤖 Initializing Agents...")
    for i in range(4):
        # Initialize PPO using the Real Environment for shape inference
        model = PPO(
            "MlpPolicy",
            env=env,                # <--- PASS REAL ENV HERE
            learning_rate=3e-4,
            n_steps=n_steps,        
            batch_size=32768,       # Large GPU batch to crunch 1M samples
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=f"logs/multi_agent_{agent_roles[i]}",
            policy_kwargs=dict(net_arch=[256, 256])
        )
        
        # --- MANUAL LOGGER SETUP ---
        # Since we use a manual loop, we must configure logger explicitly
        # otherwise model.train() will crash trying to record stats.
        new_logger = configure(f"logs/multi_agent_{agent_roles[i]}", ["stdout", "tensorboard"])
        model.set_logger(new_logger)
        
        # Reset buffer to ensure clean start
        model.rollout_buffer.reset()
        
        agents.append(model)

    # 3. Custom Training Loop
    num_updates = int(total_timesteps / total_samples_per_update) + 1
    obs = env.reset() # Shape: (8, 4, 7)
    
    print(f"🔥 Starting Loop: {num_updates} Mega-Updates Total")
    
    for update in range(1, num_updates + 1):
        # --- A. COLLECTION PHASE ---
        pbar = tqdm(range(n_steps), desc=f"Mega-Update {update}/{num_updates}", unit="step")
        start_time = time.time()
        
        for step in pbar:
            
            # 1. Get Actions from all 4 Agents
            actions_list = []
            values_list = []
            log_probs_list = []
            
            for i in range(4):
                # obs shape: (8, 4, 7) -> Slice for agent i: (8, 7)
                agent_obs = obs[:, i, :] 
                
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(agent_obs).to(agents[i].device)
                    actions, values, log_probs = agents[i].policy(obs_tensor)
                
                actions_list.append(actions.cpu().numpy())
                values_list.append(values)
                log_probs_list.append(log_probs)

            # Stack actions: (4, 8, 2) -> (8, 4, 2) for Env
            env_actions = np.stack(actions_list, axis=1)
            
            # 2. Step Environment
            new_obs, rewards, dones, infos = env.step(env_actions)
            
            # 3. Store Data in Buffers
            for i in range(4):
                agent_obs = obs[:, i, :]
                agents[i].rollout_buffer.add(
                    agent_obs,
                    actions_list[i],
                    rewards[:, i],
                    dones,
                    values_list[i],
                    log_probs_list[i]
                )
            
            obs = new_obs
            
            # Update PBar FPS
            if step % 100 == 0 and step > 0:
                elapsed = time.time() - start_time
                fps = int((step * REAL_ENVS) / elapsed)
                pbar.set_postfix(fps=fps)
        
        pbar.close()

        # --- B. UPDATE PHASE ---
        print(f"   > Optimizing Agents on {total_samples_per_update:,} samples...")
        
        for i in range(4):
            # Compute GAE
            with torch.no_grad():
                last_obs = obs[:, i, :]
                last_obs_tensor = torch.as_tensor(last_obs).to(agents[i].device)
                last_values = agents[i].policy.predict_values(last_obs_tensor)
                
            agents[i].rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
            
            # Train
            agents[i].train()
            agents[i].rollout_buffer.reset() 

        total_cycle_time = time.time() - start_time
        fps = int(total_samples_per_update / total_cycle_time)
        print(f"   > Cycle finished in {total_cycle_time:.1f}s (FPS: {fps})")
        print("-" * 60)
        
        # Save Checkpoints
        save_dir = f"saves/ws9/update_{update}"
        os.makedirs(save_dir, exist_ok=True)
        for i in range(4):
            agents[i].save(f"{save_dir}/{agent_roles[i]}")
                
    print("✅ Training Complete!")
    env.close()

if __name__ == "__main__":
    main()
