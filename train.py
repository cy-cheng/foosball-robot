import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from foosball_env import FoosballEnv
import argparse
from datetime import datetime


class CurriculumCallback(BaseCallback):
    """
    Callback to advance curriculum level based on goals scored.
    Switches to next level after N goals in current level.
    """
    def __init__(self, goals_per_level=10, verbose=0):
        super(CurriculumCallback, self).__init__(verbose=verbose)
        self.goals_per_level = goals_per_level
        self.last_level = 1
        
    def _on_step(self) -> bool:
        # Check if env is single env or vectorized
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
        else:
            env = self.training_env
        
        # Advance curriculum
        if hasattr(env, 'goals_this_level') and hasattr(env, 'curriculum_level'):
            if env.goals_this_level >= self.goals_per_level and env.curriculum_level < 4:
                env.curriculum_level += 1
                if self.verbose > 0:
                    print(f"\n🎓 CURRICULUM ADVANCED to Level {env.curriculum_level}!")
                self.last_level = env.curriculum_level
        
        return True


class StatisticsCallback(BaseCallback):
    """Log episode statistics"""
    def __init__(self, verbose=0):
        super(StatisticsCallback, self).__init__(verbose=verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.goals_scored = []
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0 and self.verbose > 0:
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
            else:
                env = self.training_env
            
            if hasattr(env, 'curriculum_level'):
                print(f"Timestep {self.num_timesteps}: Level {env.curriculum_level}, Goals: {env.goals_this_level}")
        
        return True


def make_env(rank, curriculum_level=1, player_id=1, seed=None):
    """Create a single environment"""
    def _init():
        env = FoosballEnv(
            render_mode='human' if rank == 0 else 'human',
            curriculum_level=curriculum_level,
            debug_mode=False,
            player_id=player_id
        )
        if seed is not None:
            env.reset(seed=seed + rank)
        return env
    return _init


def train(
    total_steps=1_000_000,
    curriculum_level=1,
    player_id=1,
    num_envs=1,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=2048,
    render=False,
    checkpoint_freq=50_000,
    model_name="foosball_model"
):
    """
    Train a foosball agent using PPO with curriculum learning.
    
    Args:
        total_steps: Total timesteps to train
        curriculum_level: Starting curriculum level (1-4)
        player_id: Which team to train (1 or 2)
        num_envs: Number of parallel environments
        learning_rate: PPO learning rate
        batch_size: Mini-batch size
        n_steps: Steps to collect per env before update
        render: Whether to render first environment
        checkpoint_freq: Save checkpoint every N steps
        model_name: Name for saved models
    """
    
    # Create log directories
    os.makedirs("saves", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🤖 Starting Foosball RL Training")
    print(f"   Player ID: {player_id}")
    print(f"   Starting Level: {curriculum_level}")
    print(f"   Total Steps: {total_steps:,}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Parallel Envs: {num_envs}")
    print(f"   Log Dir: {log_dir}")
    print()
    
    # Create vectorized environment
    env_fns = [
        make_env(rank=i, curriculum_level=curriculum_level, player_id=player_id, seed=42)
        for i in range(num_envs)
    ]
    env = DummyVecEnv(env_fns)
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="saves/",
        name_prefix=f"{model_name}_ckpt"
    )
    
    curriculum_callback = CurriculumCallback(goals_per_level=10, verbose=1)
    stats_callback = StatisticsCallback(verbose=1)
    
    # Train
    print("🚀 Training started...\n")
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, curriculum_callback, stats_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Save final model
    final_model_path = f"saves/{model_name}_final"
    model.save(final_model_path)
    print(f"\n✅ Final model saved to {final_model_path}.zip")
    
    env.close()
    
    return model


def evaluate(model_path, num_episodes=10, curriculum_level=4, render=True):
    """
    Evaluate a trained model in symmetric two-agent play.
    Both Team 1 and Team 2 use the same policy (mirrored).
    """
    print(f"\n🔍 Evaluating model: {model_path}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Curriculum Level: {curriculum_level}")
    print()
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment (player_id doesn't matter for eval, just use 1)
    env = FoosballEnv(
        render_mode='human' if render else 'human',
        curriculum_level=curriculum_level,
        debug_mode=False,
        player_id=1
    )
    
    episode_rewards = []
    episode_lengths = []
    total_goals = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Both agents use same policy
            action1, _ = model.predict(obs, deterministic=True)
            
            # Mirror action for team 2
            mirrored_action = action1.copy()
            mirrored_action[:4] = action1[:4]  # Slides stay same
            mirrored_action[4:] = -action1[4:]  # Rotates mirrored
            
            obs, reward, terminated, truncated, _ = env.step(action1, opponent_action=mirrored_action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_goals += env.goals_this_level
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Length={episode_length}, Goals={env.goals_this_level}")
    
    env.close()
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   Avg Length: {np.mean(episode_lengths):.1f}")
    print(f"   Total Goals: {total_goals}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train foosball RL agent")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Mode: train or eval")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--level", type=int, default=1, help="Starting curriculum level (1-4)")
    parser.add_argument("--player", type=int, choices=[1, 2], default=1, help="Player to train (1 or 2)")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per env per update")
    parser.add_argument("--model", default="foosball_model", help="Model name")
    parser.add_argument("--checkpoint", default="saves/foosball_model_final.zip", help="Model checkpoint to load")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(
            total_steps=args.steps,
            curriculum_level=args.level,
            player_id=args.player,
            num_envs=args.num_envs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            model_name=args.model,
            render=not args.no_render
        )
    else:
        evaluate(
            model_path=args.checkpoint,
            num_episodes=args.eval_episodes,
            curriculum_level=4,
            render=not args.no_render
        )
