import gymnasium as gym
from stable_baselines3 import PPO
from foosball_env import FoosballEnv

# Create the environment
env = FoosballEnv(gui=True)

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the agent
model.save("ppo_foosball")

# Close the environment
env.close()
