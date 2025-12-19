import numpy as np
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from foosball_env import FoosballEnv
from foosball_utils import load_config

def main():
    # Load Config
    config = load_config("configs/ws9.yaml")
    
    # --- SINGLE AGENT DEBUG SETUP ---
    # 1 Environment, GUI Mode, Training ONLY Striker (Agent 3)
    print("🚀 Launching Single-Agent Visual Training (Forward/Striker)...")
    
    env = FoosballEnv(config, render_mode='human', curriculum_level=1, fixed_active_agent=3)
    
    # Initialize PPO for the Forward Agent (Striker)
    model = PPO(
        "MlpPolicy",
        env=env,       
        learning_rate=3e-4,
        n_steps=2048,  
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log="logs/debug_striker"
    )
    
    # --- FIX: MANUALLY SETUP LOGGER ---
    # Since we aren't calling .learn(), we must configure the logger ourselves
    new_logger = configure("logs/debug_striker", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Setup internals for manual loop
    model.policy = model.policy.to("cpu")
    
    # Reset Buffer
    model.rollout_buffer.reset()
    
    # Training Loop
    episodes = 0
    obs, _ = env.reset() # Returns (4, 7) for 4 agents
    
    # Target Agent Index: 3 (Striker)
    target_idx = 3
    
    print("🎥 Starting Rendering Loop...")
    
    while True:
        # 1. Get Action for Striker
        # Extract just the Striker's observation (Shape: 7,)
        striker_obs = obs[target_idx, :] 
        
        with torch.no_grad():
            # Add batch dim: (1, 7)
            obs_tensor = torch.as_tensor(striker_obs).unsqueeze(0).to(model.device)
            action, value, log_prob = model.policy.forward(obs_tensor)
            action_np = action.cpu().numpy()[0] # (2,)
        
        # 2. Construct Full Action (Others doing nothing)
        full_actions = np.zeros((4, 2))
        full_actions[target_idx] = action_np 
        
        # 3. Step Environment (Manual Step)
        new_obs, rewards, terminated, truncated, _ = env.step(full_actions)
        
        # 4. Store Data for Striker
        model.rollout_buffer.add(
            striker_obs.reshape(1, -1), # Obs
            action.cpu().numpy(),       # Action
            np.array([rewards[target_idx]]), # Reward
            np.array([terminated]),     # Done
            value,
            log_prob
        )
        
        obs = new_obs
        
        # 5. Check Termination
        if terminated or truncated:
            episodes += 1
            print(f"Episode {episodes} Finished. Reward: {rewards[target_idx]:.2f}")
            obs, _ = env.reset()
            
        # 6. Update Model if Buffer Full
        if model.rollout_buffer.full:
            print("🔄 Updating Striker Policy...")
            
            # Compute GAE
            with torch.no_grad():
                last_obs = obs[target_idx, :]
                last_value = model.policy.predict_values(torch.as_tensor(last_obs).unsqueeze(0).to(model.device))
            
            model.rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=np.array([terminated]))
            
            model.train()
            model.rollout_buffer.reset()
            print("✅ Update Complete.")

if __name__ == "__main__":
    main()
