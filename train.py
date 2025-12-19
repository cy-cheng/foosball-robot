import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure

from foosball_env import FoosballEnv
from foosball_utils import load_config

def make_env(rank, config):
    def _init():
        return FoosballEnv(config, render_mode='direct', curriculum_level=1)
    return _init

def run_verification(agents, eval_env, num_episodes=5):
    total_rewards = np.zeros(4)
    goals_scored = 0
    
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = np.zeros(4)
        
        while not done:
            actions = []
            for i in range(4):
                agent_obs = obs[i]
                with torch.no_grad():
                    act, _ = agents[i].predict(agent_obs, deterministic=True)
                actions.append(act)
            
            obs, rewards, term, trunc, _ = eval_env.step(np.stack(actions))
            episode_reward += rewards
            if term or trunc:
                done = True
                if rewards[3] > 5.0: 
                    goals_scored += 1
                    
        total_rewards += episode_reward

    avg_rewards = total_rewards / num_episodes
    win_rate = goals_scored / num_episodes
    return avg_rewards, win_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ws9.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    # --- CONFIGURATION ---
    VIRTUAL_ENVS = 2048
    REAL_ENVS = 8  
    STEPS_PER_VIRTUAL_ENV = 512 
    
    SAVE_FREQ = 1         
    VERIFY_FREQ = 1       
    N_VERIFY_GAMES = 5    
    
    n_steps = int(STEPS_PER_VIRTUAL_ENV * (VIRTUAL_ENVS / REAL_ENVS))
    total_samples_per_update = n_steps * REAL_ENVS
    total_timesteps = 100_000_000
    
    print(f"🚀 Launching Massive-Batch Training")
    print(f"   - Virtual Envs: {VIRTUAL_ENVS}")
    print(f"   - Real Processes: {REAL_ENVS}")
    print(f"   - Collection Steps per Cycle: {n_steps}")
    print(f"   - Total Buffer Size: {total_samples_per_update:,} samples")

    env = SubprocVecEnv([make_env(i, config) for i in range(REAL_ENVS)])
    eval_env = FoosballEnv(config, render_mode='direct', curriculum_level=1)
    
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
        
        new_logger = configure(f"logs/multi_agent_{agent_roles[i]}", ["stdout", "tensorboard"])
        model.set_logger(new_logger)
        model.rollout_buffer.reset()
        agents.append(model)

    num_updates = int(total_timesteps / total_samples_per_update) + 1
    obs = env.reset() 

    # --- NEW: SAVE INITIAL MODEL (Update 0) ---
    # Saves the random/untrained agents immediately so you can test the pipeline.
    init_save_dir = "saves/ws9/update_0"
    os.makedirs(init_save_dir, exist_ok=True)
    print(f"   💾 Saving Initial (Random) Model to: {init_save_dir}")
    for i in range(4):
        agents[i].save(f"{init_save_dir}/{agent_roles[i]}")
    
    print(f"🔥 Starting Loop: {num_updates} Mega-Updates Total")
    
    for update in range(1, num_updates + 1):
        pbar = tqdm(range(n_steps), desc=f"Mega-Update {update}/{num_updates}", unit="step")
        start_time = time.time()
        
        goals_collected = 0
        
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
            
            # Extract goals from infos
            for info in infos:
                goals_collected += info.get('goal_scored', 0)
            
            for i in range(4):
                agents[i].rollout_buffer.add(
                    obs[:, i, :], actions_list[i], rewards[:, i], dones, values_list[i], log_probs_list[i]
                )
            
            obs = new_obs
            
            if step % 100 == 0 and step > 0:
                elapsed = time.time() - start_time
                fps = int((step * REAL_ENVS) / elapsed)
                pbar.set_postfix(fps=fps, goals=goals_collected)
        
        pbar.close()

        print(f"   > Optimizing Agents on {total_samples_per_update:,} samples...")
        print(f"   > Total Goals in Batch: {goals_collected}")
        
        for i in range(4):
            agents[i].num_timesteps += total_samples_per_update
            
            # Log training goals to TensorBoard
            agents[i].logger.record("train/goals_collected", goals_collected)
            
            with torch.no_grad():
                last_obs = obs[:, i, :]
                last_obs_tensor = torch.as_tensor(last_obs).to(agents[i].device)
                last_values = agents[i].policy.predict_values(last_obs_tensor)
            agents[i].rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
            
            agents[i].train()
            agents[i].rollout_buffer.reset() 

        if update % VERIFY_FREQ == 0:
            print(f"   > Running {N_VERIFY_GAMES} Verification Games...")
            avg_rewards, win_rate = run_verification(agents, eval_env, num_episodes=N_VERIFY_GAMES)
            
            print(f"     [EVAL] Win Rate: {win_rate*100:.1f}% | Rewards: GK:{avg_rewards[0]:.1f} DEF:{avg_rewards[1]:.1f} MID:{avg_rewards[2]:.1f} STR:{avg_rewards[3]:.1f}")
            
            for i in range(4):
                agents[i].logger.record("eval/mean_reward", avg_rewards[i])
                agents[i].logger.record("eval/win_rate_approx", win_rate)
                agents[i].logger.dump(step=agents[i].num_timesteps)
        else:
            for i in range(4):
                agents[i].logger.dump(step=agents[i].num_timesteps)

        total_cycle_time = time.time() - start_time
        fps = int(total_samples_per_update / total_cycle_time)
        print(f"   > Cycle finished in {total_cycle_time:.1f}s (FPS: {fps})")
        
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
