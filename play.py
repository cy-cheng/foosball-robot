import gymnasium as gym
from stable_baselines3 import PPO
from foosball_env import FoosballEnv

def play_model():
    # Create the environment with GUI
    env = FoosballEnv(gui=True)

    # Load the trained agent
    model = PPO.load("saves/rl_model_4921000_steps.zip", env=env)

    # Run the simulation
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    play_model()
