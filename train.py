import os
import argparse
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from foosball_env import FoosballEnv
from foosball_utils import load_config, get_config_value

# Load default configuration
DEFAULT_CONFIG_PATH = "configs/default.yaml"
if not os.path.exists(DEFAULT_CONFIG_PATH):
    raise FileNotFoundError(f"Default configuration file not found at {DEFAULT_CONFIG_PATH}")
DEFAULT_CONFIG = load_config(DEFAULT_CONFIG_PATH)


class SelfPlayCallback(BaseCallback):
    def __init__(self, update_freq: int, verbose=0):
        super(SelfPlayCallback, self).__init__(verbose)
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self.training_env.env_method("update_opponent_model", self.model.policy.to("cpu").state_dict())
        return True


class RewardLoggerCallback(BaseCallback):
    """
    A custom callback to log mean episodic reward to the console.
    """
    def __init__(self, check_freq: int, verbose: int = 1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # The ep_info_buffer is a deque of dicts {'r': reward, 'l': length, 't': timestamp}
            if hasattr(self.training_env, 'ep_info_buffer'):
                ep_info_buffer = self.training_env.ep_info_buffer
                if ep_info_buffer:
                    # Calculate the mean reward from all episodes in the buffer
                    mean_reward = sum([info['r'] for info in ep_info_buffer]) / len(ep_info_buffer)
                    if self.verbose > 0:
                        print(f"Timestep: {self.num_timesteps}, Mean Episode Reward: {mean_reward:.2f}")
        return True


def train_stage(stage, full_config, load_checkpoint=None):
    """Train a specific curriculum stage."""
    
    stage_config = full_config['curriculum'][f'stage_{stage}']
    
    # Extract common training parameters
    training_config = full_config['training']
    learning_rate = float(training_config['learning_rate'])
    num_envs = training_config['num_parallel_envs']
    checkpoint_freq = training_config['checkpoint_freq']
    num_envs_render = training_config['num_envs_render']

    # Ensure num_envs is at least num_envs_render if rendering is enabled
    if num_envs_render > 0 and num_envs < num_envs_render:
        print(f"Warning: num_parallel_envs ({num_envs}) is less than num_envs_render ({num_envs_render}). Setting num_parallel_envs to num_envs_render.")
        num_envs = num_envs_render
    
    # Extract stage-specific parameters
    stage_steps = stage_config['duration_steps']
    max_episode_steps = stage_config.get('max_episode_steps', 2000) # Default if not specified, though it should be in config
    
    # PPO hyperparameters (common for all stages unless overridden)
    ppo_params = {
        "policy": "MlpPolicy",
        "learning_rate": learning_rate,
        "batch_size": training_config['batch_size'],
        "n_steps": training_config['n_steps'],
        "n_epochs": training_config['n_epochs'], # Corrected typo
        "gamma": training_config['gamma'],
        "gae_lambda": training_config['gae_lambda'],
        "clip_range": training_config['clip_range'],
        "ent_coef": training_config['entropy_coefficient'],
        "verbose": 1,
    }

    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  Stage {stage}: {stage_config['name']}
║  {stage_config['description']}
╚════════════════════════════════════════════════════════════════════╝

  Focus: {stage_config.get('focus', 'N/A')}
  Steps: {stage_steps:,}
  Parallel Envs: {num_envs}
  Learning Rate: {learning_rate}
"""
)
    
    # Create directories
    os.makedirs("saves", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/stage_{stage}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="saves/",
        name_prefix=f"stage_{stage}_ckpt"
    )
    reward_logger_callback = RewardLoggerCallback(check_freq=1000)

    if stage == 4:
        print("Setting up self-play for Stage 4...")
        
        def make_env(rank, _num_envs_render_arg=num_envs_render): # Explicitly pass num_envs_render
            def _init():
                env_render_mode = 'human' if (_num_envs_render_arg > 0 and rank < _num_envs_render_arg) else 'direct'
                return FoosballEnv(
                    config=full_config,
                    render_mode=env_render_mode,
                    curriculum_level=stage,
                    opponent_model=None, # Will be set later
                )
            return _init

        env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(num_envs)]))

        if load_checkpoint and os.path.exists(load_checkpoint):
            print(f"✅ Loading checkpoint for agent and opponent: {load_checkpoint}")
            model = PPO.load(load_checkpoint, env=env)
            opponent_model = PPO.load(load_checkpoint)
        else:
            print("🆕 No valid checkpoint provided for Stage 4. Creating a new model from scratch.")
            model_params = ppo_params.copy()
            model_params["tensorboard_log"] = log_dir
            model = PPO(env=env, **model_params)
            
            opponent_model_params = ppo_params.copy()
            opponent_model_params.pop("policy", None)
            opponent_model = PPO(
                policy=model.policy.__class__, 
                env=None, 
                _init_setup_model=False,
                **opponent_model_params
            )
            opponent_model.observation_space = model.observation_space
            opponent_model.action_space = model.action_space
            opponent_model.n_envs = model.n_envs
            opponent_model._setup_model()

        env.env_method("update_opponent_model", opponent_model.policy.state_dict(), indices=range(num_envs))
        
        self_play_callback = SelfPlayCallback(update_freq=100_000) # update_freq could be made configurable
        
        print(f"\n🚀 Training Stage {stage} with self-play...\n")
        try:
            model.learn(
                total_timesteps=stage_steps,
                callback=[checkpoint_callback, self_play_callback, reward_logger_callback],
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted!")

    else: # Stages 1, 2, 3
        def make_env(rank, _num_envs_render_arg=num_envs_render): # Explicitly pass num_envs_render
            def _init():
                env_render_mode = 'human' if (_num_envs_render_arg > 0 and rank < _num_envs_render_arg) else 'direct'
                return FoosballEnv(
                    config=full_config,
                    render_mode=env_render_mode,
                    curriculum_level=stage
                )
            return _init
        
        print("Creating environments...")
        env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(num_envs)]))
        
        if load_checkpoint and os.path.exists(load_checkpoint):
            print(f"✅ Loading checkpoint: {load_checkpoint}")
            model = PPO.load(load_checkpoint)
            model.set_env(env)
            print(f"   Model loaded and environment set!")
        else:
            print(f"🆕 Creating new model for Stage {stage}")
            model_params = ppo_params.copy()
            model_params["tensorboard_log"] = log_dir
            model = PPO(env=env, **model_params)
        
        print(f"\n🚀 Training Stage {stage}...\n")
        try:
            model.learn(
                total_timesteps=stage_steps,
                callback=[checkpoint_callback, reward_logger_callback],
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted!")
    
    stage_checkpoint = f"saves/foosball_stage_{stage}_completed"
    model.save(stage_checkpoint)
    
    print(f"""

✅ Stage {stage} Complete!
   Saved: {stage_checkpoint}.zip
   Logs: {log_dir}/
"""
)
    
    env.close()
    
    if stage < 4:
        print(f"Next: uv run train.py --stage {stage+1} --load {stage_checkpoint}.zip --config <your_config.yaml>")
    else:
        print("🎉 All stages complete! Model ready for deployment.")
        print(f"   Final model: {stage_checkpoint}.zip")



def main():
    parser = argparse.ArgumentParser(
        description="Train foosball agent stage-by-stage with curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training configuration YAML file"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        help="Curriculum stage to train (1-4)"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all training stages sequentially"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Checkpoint to load (e.g., saves/foosball_stage_1_completed.zip)"
    )
    
    args = parser.parse_args()

    full_config = load_config(args.config)
    
    if args.run_all or (not args.stage and not args.run_all): # If no stage is specified, run all
        # Run all stages sequentially
        checkpoint = args.load
        for stage in range(1, 5):
            train_stage(
                stage=stage,
                full_config=full_config,
                load_checkpoint=checkpoint,
            )
            checkpoint = f"saves/foosball_stage_{stage}_completed.zip"
    elif args.stage:
        # Run a single stage
        train_stage(
            stage=args.stage,
            full_config=full_config,
            load_checkpoint=args.load,
        )
    else:
        parser.print_help()



if __name__ == "__main__":
    main()


