#!/usr/bin/env python3
"""
Stage-by-stage training with checkpoint loading.

Usage:
  # Stage 1 (fresh start)
  uv run train_stages.py --stage 1
  
  # Stage 2 (load Stage 1)
  uv run train_stages.py --stage 2 --load saves/foosball_stage_1_completed.zip
  
  # Stage 3 (load Stage 2)
  uv run train_stages.py --stage 3 --load saves/foosball_stage_2_completed.zip
  
  # Stage 4 (load Stage 3)
  uv run train_stages.py --stage 4 --load saves/foosball_stage_3_completed.zip
"""

import os
import argparse
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from foosball_env import FoosballEnv


class SelfPlayCallback(BaseCallback):
    def __init__(self, update_freq: int, verbose=0):
        super(SelfPlayCallback, self).__init__(verbose)
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self.training_env.env_method("update_opponent_model", self.model.policy.to("cpu").state_dict())
        return True


STAGE_INFO = {
    1: {
        "name": "Dribble",
        "description": "Ball stationary - learn basic hitting",
        "steps": 1_000_000,
        "focus": "Basic control, ball contact"
    },
    2: {
        "name": "Pass",
        "description": "Ball rolling toward you - learn interception",
        "steps": 1_000_000,
        "focus": "Interception, positioning"
    },
    3: {
        "name": "Defend",
        "description": "Ball shot fast at goal - learn defense",
        "steps": 1_000_000,
        "focus": "Blocking, defensive positioning"
    },
    4: {
        "name": "Full Game",
        "description": "Random play - learn complete strategy",
        "steps": 20_000_000,
        "steps_per_episode": 50_000,
        "focus": "Offense, defense, strategy"
    }
}


def train_stage(stage, load_checkpoint=None, steps=250_000, num_envs=4, 
                learning_rate=3e-4, render=False, steps_per_episode=2000):
    """Train a specific curriculum stage."""
    
    info = STAGE_INFO[stage]
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  Stage {stage}: {info['name']}
║  {info['description']}
╚════════════════════════════════════════════════════════════════════╝

  Focus: {info['focus']}
  Steps: {steps:,}
  Parallel Envs: {num_envs}
  Learning Rate: {learning_rate}
""")
    
    # Create directories
    os.makedirs("saves", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/stage_{stage}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="saves/",
        name_prefix=f"stage_{stage}_ckpt"
    )

    if stage == 4:
        print("Setting up self-play for Stage 4...")
        
        # Load the model from stage 3 as the main agent
        model = PPO.load(load_checkpoint)
        
        # Create a separate opponent model
        opponent_model = PPO.load(load_checkpoint)

        def make_env(rank):
            def _init():
                env_render_mode = 'human' if (rank == 0 and render) else 'direct'
                return FoosballEnv(
                    render_mode=env_render_mode,
                    curriculum_level=stage,
                    opponent_model=opponent_model,
                    steps_per_episode=steps_per_episode
                )
            return _init
            
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
        model.set_env(env)
        
        # Callback for self-play
        self_play_callback = SelfPlayCallback(update_freq=100_000)
        
        # Train
        print(f"\n🚀 Training Stage {stage} with self-play...\n")
        try:
            model.learn(
                total_timesteps=steps,
                callback=[checkpoint_callback, self_play_callback],
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted!")

    else: # Stages 1, 2, 3
        def make_env(rank):
            def _init():
                env_render_mode = 'human' if (rank == 0 and render) else 'direct'
                return FoosballEnv(
                    render_mode=env_render_mode,
                    curriculum_level=stage
                )
            return _init
        
        print("Creating environments...")
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
        
        # Load or create model
        if load_checkpoint and os.path.exists(load_checkpoint):
            print(f"✅ Loading checkpoint: {load_checkpoint}")
            model = PPO.load(load_checkpoint)
            model.set_env(env)
            print(f"   Model loaded and environment set!")
        else:
            print(f"🆕 Creating new model for Stage {stage}")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                batch_size=64,
                n_steps=2048,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=log_dir
            )
        
        # Train
        print(f"\n🚀 Training Stage {stage}...\n")
        try:
            model.learn(
                total_timesteps=steps,
                callback=checkpoint_callback,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted!")
    
    # Save stage checkpoint
    stage_checkpoint = f"saves/foosball_stage_{stage}_completed"
    model.save(stage_checkpoint)
    
    print(f"""
✅ Stage {stage} Complete!
   Saved: {stage_checkpoint}.zip
   Logs: {log_dir}/
""")
    
    env.close()
    
    if stage < 4:
        print(f"Next: uv run train.py --stage {stage+1} --load {stage_checkpoint}.zip")
    else:
        print("🎉 All stages complete! Model ready for deployment.")
        print(f"   Final model: {stage_checkpoint}.zip")


def main():
    parser = argparse.ArgumentParser(
        description="Train foosball agent stage-by-stage with curriculum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Steps to train (default: 250K per stage)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable GUI rendering"
    )
    
    args = parser.parse_args()

    if args.run_all or len(sys.argv) == 1:
        # Run all stages sequentially
        checkpoint = None
        for stage in range(1, 5):
            steps = args.steps if args.steps else STAGE_INFO[stage]["steps"]
            steps_per_episode = STAGE_INFO[stage].get("steps_per_episode", 2000)
            train_stage(
                stage=stage,
                load_checkpoint=checkpoint,
                steps=steps,
                num_envs=args.num_envs,
                learning_rate=args.lr,
                render=args.render,
                steps_per_episode=steps_per_episode
            )
            checkpoint = f"saves/foosball_stage_{stage}_completed.zip"
    elif args.stage:
        # Run a single stage
        steps = args.steps if args.steps else STAGE_INFO[args.stage]["steps"]
        train_stage(
            stage=args.stage,
            load_checkpoint=args.load,
            steps=steps,
            num_envs=args.num_envs,
            learning_rate=args.lr,
            render=args.render
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
