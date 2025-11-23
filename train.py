import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from foosball_env import FoosballEnv

# Create the environment
env = FoosballEnv(gui=True)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model')

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1, device='cpu')

# Train the agent
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save the agent
model.save("ppo_foosball")

# Close the environment
env.close()
