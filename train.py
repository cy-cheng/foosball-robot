import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from foosball_env import FoosballEnv
from foosball_utils import load_config

# --- HELPER CLASSES ---

def make_env(rank, config):
    """Factory for SubprocVecEnv"""
    def _init():
        return FoosballEnv(config, render_mode='direct', curriculum_level=1)
    return _init

def run_verification(agents, eval_env, num_episodes=5):
    """
    Runs verification games to evaluate the current policy.
    Returns: avg_rewards (per agent), approximate_win_rate
    """
    total_rewards = np.zeros(4)
    goals_scored = 0
    
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = np.zeros(4)
        
        while not done:
            actions = []
            # Get deterministic actions from all 4 agents
            for i in range(4):
                agent_obs = obs[i]
                with torch.no_grad():
                    # predict() returns (action, state)
                    act, _ = agents[i].predict(agent_obs, deterministic=True)
                actions.append(act)
            
            # Step environment
            obs, rewards, term, trunc, _ = eval_env.step(np.stack(actions))
            episode_reward += rewards
            
            if term or trunc:
                done = True
                # Heuristic: If Striker (idx 3) got a large reward (>5), it likely scored/touched goal
                if rewards[3] > 5.0: 
                    goals_scored += 1
                    
        total_rewards += episode_reward

    avg_rewards = total_rewards / num_episodes
    win_rate = goals_scored / num_episodes
    return avg_rewards, win_rate

# --- MAIN TRAINING LOOP ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ws9.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    # --- CONFIGURATION ---
    # Goal: Simulate 2048 Environments using 8 Processes (Accumulation)
    VIRTUAL_ENVS = 128
    REAL_ENVS = 8  
    STEPS_PER_VIRTUAL_ENV = 128
    
    # Checkpoint & Verification Settings
    SAVE_FREQ = 1         # Save every X mega-updates
    VERIFY_FREQ = 1       # Run verification games every X mega-updates
    N_VERIFY_GAMES = 5    # How many games to play for verification
    
    # Calculate accumulation steps
    n_steps = int(STEPS_PER_VIRTUAL_ENV * (VIRTUAL_ENVS / REAL_ENVS))
    
    total_samples_per_update = n_steps * REAL_ENVS
    total_timesteps = 100_000
    
    print(f"🚀 Launching Massive-Batch Training")
    print(f"   - Virtual Envs: {VIRTUAL_ENVS}")
    print(f"   - Real Processes: {REAL_ENVS}")
    print(f"   - Collection Steps per Cycle: {n_steps}")
    print(f"   - Total Buffer Size: {total_samples_per_update:,} samples")

    # 1. Create Real Environment (Training)
    env = SubprocVecEnv([make_env(i, config) for i in range(REAL_ENVS)])
    
    # 2. Create Evaluation Environment (Single instance for verification)
    eval_env = FoosballEnv(config, render_mode='direct', curriculum_level=1)
    
    # 3. Initialize 4 Agents (GK, DEF, MID, STR)
    agent_roles = ["GK", "DEF", "MID", "STR"]
    agents = []
    
    print("🤖 Initializing Agents...")
    for i in range(4):
        model = PPO(
            "MlpPolicy",
            env=env,                
            learning_rate=3e-4,
            n_steps=n_steps,        
            batch_size=32768,       
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=f"logs/multi_agent_{agent_roles[i]}",
            policy_kwargs=dict(net_arch=[256, 256])
        )
        
        # Manual Logger Setup
        new_logger = configure(f"logs/multi_agent_{agent_roles[i]}", ["stdout", "tensorboard"])
        model.set_logger(new_logger)
        model.rollout_buffer.reset()
        
        agents.append(model)

    # 4. Custom Training Loop
    num_updates = int(total_timesteps / total_samples_per_update) + 1
    obs = env.reset() # Shape: (8, 4, 7)
    
    print(f"🔥 Starting Loop: {num_updates} Mega-Updates Total")
    
    for update in range(1, num_updates + 1):
        # --- A. COLLECTION PHASE ---
        pbar = tqdm(range(n_steps), desc=f"Mega-Update {update}/{num_updates}", unit="step")
        start_time = time.time()
        
        for step in pbar:
            actions_list = []
            values_list = []
            log_probs_list = []
            
            for i in range(4):
                agent_obs = obs[:, i, :] 
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(agent_obs).to(agents[i].device)
                    actions, values, log_probs = agents[i].policy(obs_tensor)
                
                actions_list.append(actions.cpu().numpy())
                values_list.append(values)
                log_probs_list.append(log_probs)

            env_actions = np.stack(actions_list, axis=1)
            new_obs, rewards, dones, infos = env.step(env_actions)
            
            for i in range(4):
                agents[i].rollout_buffer.add(
                    obs[:, i, :], actions_list[i], rewards[:, i], dones, values_list[i], log_probs_list[i]
                )
            
            obs = new_obs
            
            if step % 100 == 0 and step > 0:
                elapsed = time.time() - start_time
                fps = int((step * REAL_ENVS) / elapsed)
                pbar.set_postfix(fps=fps)
        
        pbar.close()

        # --- B. UPDATE PHASE ---
        print(f"   > Optimizing Agents on {total_samples_per_update:,} samples...")
        
        for i in range(4):
            # 1. Update Timesteps for TensorBoard X-Axis
            agents[i].num_timesteps += total_samples_per_update
            
            # 2. Compute GAE
            with torch.no_grad():
                last_obs = obs[:, i, :]
                last_obs_tensor = torch.as_tensor(last_obs).to(agents[i].device)
                last_values = agents[i].policy.predict_values(last_obs_tensor)
            agents[i].rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
            
            # 3. Train (Logs Loss automatically)
            agents[i].train()
            agents[i].rollout_buffer.reset() 

        # --- C. VERIFICATION PHASE ---
        if update % VERIFY_FREQ == 0:
            print(f"   > Running {N_VERIFY_GAMES} Verification Games...")
            avg_rewards, win_rate = run_verification(agents, eval_env, num_episodes=N_VERIFY_GAMES)
            
            print(f"     [EVAL] Win Rate: {win_rate*100:.1f}% | Rewards: GK:{avg_rewards[0]:.1f} DEF:{avg_rewards[1]:.1f} MID:{avg_rewards[2]:.1f} STR:{avg_rewards[3]:.1f}")
            
            # Log Verification metrics to TensorBoard
            for i in range(4):
                agents[i].logger.record("eval/mean_reward", avg_rewards[i])
                agents[i].logger.record("eval/win_rate_approx", win_rate)
                # Dump logs to disk (flushes training loss + eval metrics)
                agents[i].logger.dump(step=agents[i].num_timesteps)
        else:
            # If skipping verification, still dump training logs
            for i in range(4):
                agents[i].logger.dump(step=agents[i].num_timesteps)

        total_cycle_time = time.time() - start_time
        fps = int(total_samples_per_update / total_cycle_time)
        print(f"   > Cycle finished in {total_cycle_time:.1f}s (FPS: {fps})")
        
        # --- D. SAVE ROUTINE ---
        if update % SAVE_FREQ == 0:
            save_dir = f"saves/ws9/update_{update}"
            os.makedirs(save_dir, exist_ok=True)
            print(f"   💾 Saving Checkpoint to: {save_dir}")
            for i in range(4):
                agents[i].save(f"{save_dir}/{agent_roles[i]}")
        
        print("-" * 60)
                
    print("✅ Training Complete!")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
