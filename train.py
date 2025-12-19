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

# --- HELPER CLASSES ---
def make_env(rank, config, initial_stage=1):
    def _init():
        return FoosballEnv(config, render_mode='direct', curriculum_level=initial_stage)
    return _init

def update_curriculum(env, level):
    env.env_method("update_stage_config", level)
    print(f"\n📢 CURRICULUM UPGRADE: Switched to Stage {level} 📢\n")

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
                with torch.no_grad():
                    act, _ = agents[i].predict(obs[i], deterministic=True)
                actions.append(act)
            obs, rewards, term, trunc, _ = eval_env.step(np.stack(actions))
            episode_reward += rewards
            if term or trunc:
                done = True
                if rewards[3] > 5.0: goals_scored += 1
        total_rewards += episode_reward
    return total_rewards / num_episodes, goals_scored / num_episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ws9.yaml")
    # CLI overrides Config if provided
    parser.add_argument("--resume", type=str, default=None) 
    parser.add_argument("--start_stage", type=int, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    train_cfg = cfg['training']
    curr_cfg = cfg['curriculum']
    
    # --- SETUP PARAMETERS ---
    VIRTUAL_ENVS = train_cfg['virtual_envs']
    REAL_ENVS = train_cfg['real_envs']
    STEPS_PER_VIRTUAL = train_cfg['steps_per_virtual']
    N_STEPS = int(STEPS_PER_VIRTUAL * (VIRTUAL_ENVS / REAL_ENVS))
    TOTAL_SAMPLES = N_STEPS * REAL_ENVS
    
    # Checkpoint Logic
    resume_path = args.resume if args.resume else train_cfg.get('resume_checkpoint')
    current_stage = args.start_stage if args.start_stage else train_cfg.get('start_stage', 1)

    print(f"🚀 Launching Flexible Training: {train_cfg['experiment_name']}")
    print(f"   - Virtual: {VIRTUAL_ENVS} | Real: {REAL_ENVS}")
    print(f"   - Cycle Steps: {N_STEPS} | Buffer: {TOTAL_SAMPLES:,}")
    print(f"   - Start Stage: {current_stage}")
    if resume_path: print(f"   - Resuming from: {resume_path}")

    # Initialize Env
    env = SubprocVecEnv([make_env(i, cfg, current_stage) for i in range(REAL_ENVS)])
    eval_env = FoosballEnv(cfg, render_mode='direct', curriculum_level=current_stage)
    
    # Initialize Agents
    agent_roles = ["GK", "DEF", "MID", "STR"]
    agents = []
    
    print("🤖 Initializing Agents...")
    for i in range(4):
        model = PPO(
            "MlpPolicy", env=env, 
            learning_rate=float(train_cfg['learning_rate']),
            n_steps=N_STEPS, batch_size=train_cfg['batch_size'], n_epochs=train_cfg['n_epochs'],
            gamma=train_cfg['gamma'], gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            verbose=1, tensorboard_log=f"logs/{train_cfg['experiment_name']}_{agent_roles[i]}",
            policy_kwargs=dict(net_arch=[256, 256])
        )
        
        # Load Weights if Resuming
        if resume_path:
            chk_path = f"{resume_path}/{agent_roles[i]}.zip"
            if os.path.exists(chk_path):
                print(f"   📥 Loading {agent_roles[i]} from {chk_path}")
                loaded = PPO.load(chk_path, env=env)
                model.policy.load_state_dict(loaded.policy.state_dict())
                model.num_timesteps = loaded.num_timesteps
            else:
                print(f"   ⚠️ Warning: {chk_path} not found. Starting fresh.")
        
        # Logger
        new_logger = configure(f"logs/{train_cfg['experiment_name']}_{agent_roles[i]}", ["stdout", "tensorboard"])
        model.set_logger(new_logger)
        model.rollout_buffer.reset()
        agents.append(model)

    # --- MAIN LOOP ---
    num_updates = int(train_cfg['total_timesteps'] / TOTAL_SAMPLES) + 1
    obs = env.reset()
    
    # Save Update 0
    if not resume_path:
        os.makedirs(f"saves/{train_cfg['experiment_name']}/update_0", exist_ok=True)
        for i in range(4): agents[i].save(f"saves/{train_cfg['experiment_name']}/update_0/{agent_roles[i]}")

    # Determine Schedule from Config
    # stage_duration: {1: 30, 2: 30, 3: -1}
    # Calculate update thresholds: Stage 2 starts at 31, Stage 3 at 61...
    stage_starts = {}
    accumulated_updates = 0
    sorted_stages = sorted([int(k.split('_')[1]) for k in curr_cfg.keys()])
    
    for s in sorted_stages:
        duration = curr_cfg[f'stage_{s}']['duration_updates']
        if duration == -1: duration = 999999
        stage_starts[s] = accumulated_updates + 1
        accumulated_updates += duration

    print(f"📅 Curriculum Schedule (Start Updates): {stage_starts}")

    start_update_idx = 1
    # Hacky: Try to guess update number from resume path
    if resume_path:
        import re
        match = re.search(r'update_(\d+)', resume_path)
        if match: start_update_idx = int(match.group(1)) + 1
    
    for update in range(start_update_idx, num_updates + 1):
        
        # Check Curriculum Switch
        # Iterate stages to see which one we should be in
        target_stage = current_stage
        for s in sorted_stages:
            if update >= stage_starts[s]:
                target_stage = s
        
        if target_stage != current_stage:
            current_stage = target_stage
            update_curriculum(env, current_stage)
            eval_env.update_stage_config(current_stage)

        pbar = tqdm(range(N_STEPS), desc=f"Mega-Update {update}/{num_updates} [Stage {current_stage}]", unit="step")
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
            for info in infos: goals_collected += info.get('goal_scored', 0)
            
            for i in range(4):
                agents[i].rollout_buffer.add(obs[:, i, :], actions_list[i], rewards[:, i], dones, values_list[i], log_probs_list[i])
            obs = new_obs
            
            if step % 100 == 0 and step > 0:
                elapsed = time.time() - start_time
                fps = int((step * REAL_ENVS) / elapsed)
                pbar.set_postfix(fps=fps, goals=goals_collected)
        pbar.close()

        print(f"   > Optimizing (Stage {current_stage} Goals: {goals_collected})")
        for i in range(4):
            agents[i].num_timesteps += TOTAL_SAMPLES
            agents[i].logger.record("train/goals_collected", goals_collected)
            agents[i].logger.record("train/curriculum_level", current_stage)
            
            with torch.no_grad():
                last_obs = obs[:, i, :]
                last_obs_tensor = torch.as_tensor(last_obs).to(agents[i].device)
                last_values = agents[i].policy.predict_values(last_obs_tensor)
            agents[i].rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
            agents[i].train()
            agents[i].rollout_buffer.reset() 

        if update % train_cfg['verify_freq'] == 0:
            avg_rewards, win_rate = run_verification(agents, eval_env, num_episodes=train_cfg['n_verify_games'])
            print(f"     [EVAL] Win Rate: {win_rate*100:.1f}% | STR Reward: {avg_rewards[3]:.1f}")
            for i in range(4):
                agents[i].logger.record("eval/mean_reward", avg_rewards[i])
                agents[i].logger.record("eval/win_rate_approx", win_rate)
                agents[i].logger.dump(step=agents[i].num_timesteps)
        else:
            for i in range(4): agents[i].logger.dump(step=agents[i].num_timesteps)

        if update % train_cfg['save_freq'] == 0:
            save_dir = f"saves/{train_cfg['experiment_name']}/update_{update}"
            os.makedirs(save_dir, exist_ok=True)
            for i in range(4): agents[i].save(f"{save_dir}/{agent_roles[i]}")
        print("-" * 60)
                
    print("✅ Training Complete!")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
