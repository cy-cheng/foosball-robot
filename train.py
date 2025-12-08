import os
import argparse
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from typing import Dict, Any

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
            # 將目前模型的策略權重，傳輸給環境中的對手模型
            self.training_env.env_method("update_opponent_model", self.model.policy.to("cpu").state_dict())
        return True

class CurriculumCallback(BaseCallback):
    """
    當達到以下雙重標準時，停止訓練（進入下一關）：
    1. 總進球數 > min_goals
    2. 勝率 (進球 / (進球+失球)) > win_rate_threshold
    """
    def __init__(self, min_goals: int, win_rate_threshold: float, check_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.min_goals = min_goals
        self.win_rate_threshold = win_rate_threshold
        self.check_freq = check_freq
        
        # 累計數據
        self.total_goals = 0
        self.total_conceded = 0
        
    def _on_step(self) -> bool:
        # 1. 累計進球與失球
        infos = self.locals.get("infos", [])
        for info in infos:
            self.total_goals += info.get("goal_scored", 0)
            self.total_conceded += info.get("goal_conceded", 0)
        
        # 2. 定期檢查
        if self.n_calls % self.check_freq == 0:
            total_games = self.total_goals + self.total_conceded
            current_win_rate = 0.0
            if total_games > 0:
                current_win_rate = self.total_goals / total_games
            
            if self.verbose > 0:
                stage_level = self.training_env.envs[0].curriculum_level if hasattr(self.training_env, 'envs') else "N/A"
                print(f"[{self.num_timesteps} steps] Stage {stage_level} Stats:")
                print(f"   - Goals: {self.total_goals} / {self.min_goals}")
                print(f"   - Win Rate: {current_win_rate:.2%} (Target: {self.win_rate_threshold:.2%})")
                print(f"   - (Scored: {self.total_goals}, Conceded: {self.total_conceded})")

            # 3. 雙重條件檢查
            if self.total_goals >= self.min_goals and current_win_rate >= self.win_rate_threshold:
                print(f"\n🎉 恭喜！達成晉級標準！")
                print(f"   - 總進球: {self.total_goals} (>= {self.min_goals})")
                print(f"   - 勝率: {current_win_rate:.2%} (>= {self.win_rate_threshold:.2%})")
                print("   -> 提早結束本關卡，準備進入下一關。")
                return False  # 停止訓練
                
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
    save_path = training_config.get('save_path')
    
    # 【修改】讀取 Curriculum 相關設定
    curriculum_enabled = training_config.get('curriculum_enabled', False)
    progression_min_goals = training_config.get('progression_min_goals', 100)
    progression_win_rate = training_config.get('progression_win_rate', 0.8)
    
    # Ensure num_envs is at least num_envs_render if rendering is enabled
    if num_envs_render > 0 and num_envs < num_envs_render:
        print(f"Warning: num_parallel_envs ({num_envs}) is less than num_envs_render ({num_envs_render}). Setting num_parallel_envs to num_envs_render.")
        num_envs = num_envs_render
    
    # Extract stage-specific parameters
    stage_steps = stage_config['duration_steps']
    max_episode_steps = stage_config.get('max_episode_steps', 2000)
    
    # 【修改】PPO policy_kwargs：加入 log_std_init 修正 std 爆炸
    policy_kwargs = dict(
        # 預設為 -1，可透過 YAML 設定檔中的 'log_std_init' 覆蓋
        log_std_init=training_config.get('log_std_init', -1) 
    )
    
    # PPO hyperparameters (common for all stages unless overridden)
    ppo_params = {
        "policy": "MlpPolicy",
        "learning_rate": learning_rate,
        "batch_size": training_config['batch_size'],
        "n_steps": training_config['n_steps'],
        "n_epochs": training_config['n_epochs'],
        "gamma": training_config['gamma'],
        "gae_lambda": training_config['gae_lambda'],
        "clip_range": training_config['clip_range'],
        "ent_coef": training_config['entropy_coefficient'],
        "verbose": 1,
        "policy_kwargs": policy_kwargs, # <--- 傳入 policy_kwargs
    }

    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  Stage {stage}: {stage_config['name']}
║  {stage_config['description']}
╚════════════════════════════════════════════════════════════════════╝

  Focus: {stage_config.get('focus', 'N/A')}
  Steps (Max): {stage_steps:,}
  Goal (Progress): {progression_min_goals:,} Goals
  Parallel Envs: {num_envs}
  Learning Rate: {learning_rate}
  Initial Log Std: {policy_kwargs.get('log_std_init')}
"""
)
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    
    # Determine save path for checkpoints and ensure it exists
    checkpoint_save_path = save_path if save_path else "saves/"
    os.makedirs(checkpoint_save_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/stage_{stage}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_save_path,
        name_prefix=f"stage_{stage}_ckpt",
        save_vecnormalize=True
    )
    
    # 【新增】Curriculum Callback 實例化
    curriculum_callback = None
    if curriculum_enabled:
        curriculum_callback = CurriculumCallback(
            min_goals=progression_min_goals,
            win_rate_threshold=progression_win_rate,
            check_freq=training_config.get('curriculum_check_freq', 5000)
        )


    # --- Environment Setup ---
    def make_env(rank, _num_envs_render_arg=num_envs_render):
        def _init():
            env_render_mode = 'human' if (_num_envs_render_arg > 0 and rank < _num_envs_render_arg) else 'direct'
            opponent_model = None  # Default for all stages
            if stage == 4 and load_checkpoint:
                # In self-play, opponent model is loaded dynamically
                pass
            
            return FoosballEnv(
                config=full_config,
                render_mode=env_render_mode,
                curriculum_level=stage,
                opponent_model=opponent_model,
            )
        return _init

    # Create the vectorized environment and wrap it with VecMonitor
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    vec_env = VecMonitor(vec_env, info_keywords=("goal_scored", "goal_conceded"))
    
    # Load normalization stats if a checkpoint is provided
    vec_normalize_path = None
    if load_checkpoint:
        vec_normalize_path = load_checkpoint.replace(".zip", "_vecnormalize.pkl")

    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"✅ Loading VecNormalize stats from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, vec_env)
    else:
        print("🆕 Creating new VecNormalize wrapper.")
        env = VecNormalize(vec_env, gamma=training_config['gamma'])

    # --- Model Setup ---
    if stage == 4:
        print("Setting up self-play for Stage 4...")
        
        # ... (Stage 4 model loading/creation logic unchanged) ...
        if load_checkpoint and os.path.exists(load_checkpoint):
            print(f"✅ Loading checkpoint for agent and opponent: {load_checkpoint}")
            model = PPO.load(load_checkpoint, env=env, **ppo_params) # 【修正】新增 ppo_params
            opponent_model = PPO.load(load_checkpoint)
        else:
            print("🆕 No valid checkpoint provided for Stage 4. Creating a new model from scratch.")
            model_params = ppo_params.copy()
            model_params["tensorboard_log"] = log_dir
            model = PPO(env=env, **model_params)
            
            # Create a shell for the opponent model
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

        # Update opponent in the environment
        env.env_method("update_opponent_model", opponent_model.policy.state_dict(), indices=range(num_envs))
        
        # Setup self-play callback
        self_play_callback = SelfPlayCallback(update_freq=100_000)
        
        # 【修改】添加 CurriculumCallback
        callbacks = [checkpoint_callback, self_play_callback]
        if curriculum_callback:
            callbacks.append(curriculum_callback)

    else: # Stages 1, 2, 3
        if load_checkpoint and os.path.exists(load_checkpoint):
            print(f"✅ Loading checkpoint: {load_checkpoint}")
            model = PPO.load(load_checkpoint, env=env, **ppo_params) # 【修正】新增 ppo_params
            print(f"   Model loaded and environment set!")
        else:
            print(f"🆕 Creating new model for Stage {stage}")
            model_params = ppo_params.copy()
            model_params["tensorboard_log"] = log_dir
            model = PPO(env=env, **model_params)
        
        # 【修改】添加 CurriculumCallback
        callbacks = [checkpoint_callback]
        if curriculum_callback:
            callbacks.append(curriculum_callback)

    # --- Training ---
    print(f"\n🚀 Training Stage {stage}...\n")
    try:
        # model.learn() 將會被 CurriculumCallback 提早中斷
        model.learn(
            total_timesteps=stage_steps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted!")

    # ... (Saving logic remains the same) ...
    if save_path:
        stage_checkpoint_path = f"{save_path}-{stage}"
    else:
        stage_checkpoint_path = f"saves/foosball_stage_{stage}_completed"
    
    model.save(stage_checkpoint_path)
    env.save(stage_checkpoint_path.replace(".zip", "") + "_vecnormalize.pkl")
    
    print(f"""

✅ Stage {stage} Complete!
   Saved Model: {stage_checkpoint_path}.zip
   Saved VecNormalize Stats: {stage_checkpoint_path}_vecnormalize.pkl
   Logs: {log_dir}/
"""
)
    
    env.close()
    
    if stage < 4:
        print(f"Next: uv run train.py --stage {stage+1} --load {stage_checkpoint_path}.zip --config <your_config.yaml>")
    else:
        print("🎉 All stages complete! Model ready for deployment.")
        print(f"   Final model: {stage_checkpoint_path}.zip")



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
    save_path = full_config['training'].get('save_path')
    
    if args.run_all or (not args.stage and not args.run_all): # If no stage is specified, run all
        # Run all stages sequentially
        checkpoint = args.load
        for stage in range(1, 5):
            # train_stage 結束時會存檔，所以下一關要讀取最新的檔
            train_stage(
                stage=stage,
                full_config=full_config,
                load_checkpoint=checkpoint,
            )
            # 更新 checkpoint path
            if save_path:
                checkpoint = f"{save_path}-{stage}.zip"
            else:
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
