import gymnasium as gym
import argparse
from stable_baselines3 import PPO
from foosball_env import FoosballEnv

def play_model(model_path):
    # Create the environment with GUI
    env = FoosballEnv(gui=True)

    # Load the trained agent
    model = PPO.load(model_path, env=env)

    # Run the simulation
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Foosball with a trained AI model")
    parser.add_argument('model', help='Path to the trained model zip file')
    args = parser.parse_args()
    play_model(args.model)
