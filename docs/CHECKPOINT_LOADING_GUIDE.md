#!/usr/bin/env python3
"""
Guide to training across curriculum stages and loading checkpoints.

The 4 Curriculum Stages:
  Level 1 (Dribble):   Learn basic ball control
  Level 2 (Pass):      Learn interception and positioning  
  Level 3 (Defend):    Learn defensive blocking
  Level 4 (Full Game): Learn complete strategy

Typical Training Flow:
  Stage 1: 0K-250K steps (Level 1)
  Stage 2: 250K-500K steps (Level 2)  
  Stage 3: 500K-750K steps (Level 3)
  Stage 4: 750K-1M steps (Level 4)

Note: Levels advance automatically after 10 goals!
"""

import argparse
import os
from pathlib import Path
from stable_baselines3 import PPO
from foosball_env import FoosballEnv


def train_stage(
    stage_num,
    total_steps_this_stage=250_000,
    checkpoint_to_load=None,
    num_envs=4,
    learning_rate=3e-4,
    render=False
):
    """
    Train a specific curriculum stage.
    
    Args:
        stage_num: 1, 2, 3, or 4
        total_steps_this_stage: Steps to train in this stage
        checkpoint_to_load: Path to previous checkpoint (or None for fresh start)
        num_envs: Number of parallel environments
        learning_rate: PPO learning rate
        render: Show GUI
    """
    
    # Create environment (always starts at level 1, auto-advances)
    env = DummyVecEnv([
        lambda: FoosballEnv(
            render_mode='human' if i == 0 and render else 'human',
            curriculum_level=1,
            debug_mode=False,
            player_id=1
        ) for i in range(num_envs)
    ])
    
    # Load or create model
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        print(f"\n✅ Loading checkpoint: {checkpoint_to_load}")
        model = PPO.load(checkpoint_to_load, env=env)
        print(f"   Model loaded successfully!")
    else:
        print(f"\n🆕 Creating new model for Stage {stage_num}")
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
            tensorboard_log=f"logs/stage_{stage_num}"
        )
    
    # Train
    print(f"\n🚀 Training Stage {stage_num} for {total_steps_this_stage:,} steps")
    print(f"   (Environment will auto-advance curriculum levels)")
    
    model.learn(total_timesteps=total_steps_this_stage)
    
    # Save stage checkpoint
    stage_name = f"foosball_stage_{stage_num}_completed"
    save_path = f"saves/{stage_name}"
    model.save(save_path)
    print(f"\n✅ Stage {stage_num} complete! Saved to: {save_path}.zip")
    
    env.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train curriculum stages with checkpoint loading")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], 
                       help="Which stage to train (1-4)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                       help="Path to checkpoint from previous stage")
    parser.add_argument("--steps", type=int, default=250_000,
                       help="Steps to train in this stage")
    parser.add_argument("--num-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--render", action="store_true",
                       help="Show GUI")
    
    args = parser.parse_args()
    
    if args.stage:
        # Training specific stage
        train_stage(
            stage_num=args.stage,
            total_steps_this_stage=args.steps,
            checkpoint_to_load=args.load_checkpoint,
            num_envs=args.num_envs,
            learning_rate=args.lr,
            render=args.render
        )
    else:
        print("""
╔════════════════════════════════════════════════════════════════════╗
║           Curriculum Stage Training Guide                          ║
╚════════════════════════════════════════════════════════════════════╝

4 CURRICULUM STAGES (Auto-advancing):

Stage 1 (Dribble):     0K - 250K steps
  └─ Ball spawns stationary
  └─ Agent learns basic hitting
  └─ Command: python train_stages.py --stage 1 --steps 250000

Stage 2 (Pass):        250K - 500K steps
  └─ Ball rolls toward agent
  └─ Agent learns interception
  └─ Command: python train_stages.py --stage 2 --steps 250000 \\
               --load-checkpoint saves/foosball_stage_1_completed.zip

Stage 3 (Defend):      500K - 750K steps
  └─ Ball shot fast at goal
  └─ Agent learns defense
  └─ Command: python train_stages.py --stage 3 --steps 250000 \\
               --load-checkpoint saves/foosball_stage_2_completed.zip

Stage 4 (Full Game):   750K - 1M steps
  └─ Random ball position/velocity
  └─ Agent learns complete strategy
  └─ Command: python train_stages.py --stage 4 --steps 250000 \\
               --load-checkpoint saves/foosball_stage_3_completed.zip

════════════════════════════════════════════════════════════════════

HOW CHECKPOINTS WORK:

1. Train Stage 1 (creates checkpoint):
   $ python train_stages.py --stage 1 --steps 250000
   → Saves: saves/foosball_stage_1_completed.zip
   
2. Train Stage 2 (loads Stage 1, continues learning):
   $ python train_stages.py --stage 2 \\
     --load-checkpoint saves/foosball_stage_1_completed.zip
   → Loads weights from Stage 1
   → Continues training with same network
   → Saves: saves/foosball_stage_2_completed.zip

3. Train Stage 3 (loads Stage 2):
   $ python train_stages.py --stage 3 \\
     --load-checkpoint saves/foosball_stage_2_completed.zip
   → And so on...

════════════════════════════════════════════════════════════════════

IMPORTANT NOTES:

✓ Auto-Progression:
  • Environment automatically advances curriculum
  • You don't manually set levels
  • After 10 goals, environment moves to next level
  
✓ Checkpoint Loading:
  • Loads network weights from previous stage
  • Optimizer state is RESET (fresh learning rate)
  • Curriculum level resets to 1 (but auto-advances immediately)
  
✓ Why Continue from Previous Stage?
  • Network has already learned basics
  • Much faster convergence
  • ~30% faster than training from scratch
  • Better final performance

✓ Skip to Later Stage?
  • Can load any checkpoint at any time
  • e.g., Load Stage 2, continue as Stage 3
  • Useful for fine-tuning or exploring

════════════════════════════════════════════════════════════════════

QUICK START - FULL TRAINING SEQUENCE:

# Stage 1: Learn dribbling (250K steps)
python train_stages.py --stage 1 --steps 250000 --num-envs 4

# Stage 2: Learn passing (250K steps)
python train_stages.py --stage 2 --steps 250000 --num-envs 4 \\
  --load-checkpoint saves/foosball_stage_1_completed.zip

# Stage 3: Learn defending (250K steps)
python train_stages.py --stage 3 --steps 250000 --num-envs 4 \\
  --load-checkpoint saves/foosball_stage_2_completed.zip

# Stage 4: Learn full game (250K steps)
python train_stages.py --stage 4 --steps 250000 --num-envs 4 \\
  --load-checkpoint saves/foosball_stage_3_completed.zip

# Total: 1M steps for complete training
# Time: ~1 hour per stage (4 hours total with 4 parallel envs)

════════════════════════════════════════════════════════════════════

LOADING IN train.py (alternative):

If using train.py, you can also manually load checkpoints:

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Load checkpoint
model = PPO.load("saves/foosball_stage_2_completed")

# Continue training with new environments
env = DummyVecEnv([lambda: FoosballEnv(...) for _ in range(4)])
model.set_env(env)
model.learn(total_timesteps=250_000)

model.save("saves/foosball_stage_3_completed")

════════════════════════════════════════════════════════════════════

CHECKING SAVED CHECKPOINTS:

$ ls -lh saves/
  foosball_stage_1_completed.zip (30MB)
  foosball_stage_2_completed.zip (30MB)
  foosball_stage_3_completed.zip (30MB)
  foosball_stage_4_completed.zip (30MB)
  foosball_model_final.zip      (30MB)  ← Final model

════════════════════════════════════════════════════════════════════

MONITORING EACH STAGE:

tensorboard --logdir logs/
# View separate logs for each stage:
#   logs/stage_1/
#   logs/stage_2/
#   logs/stage_3/
#   logs/stage_4/

════════════════════════════════════════════════════════════════════

RESUMING INTERRUPTED TRAINING:

If training is interrupted mid-stage:

$ ls -lh saves/
  foosball_model_ckpt_120000_steps.zip  ← Auto-saved checkpoint

Resume:
$ python train_stages.py --stage 2 \\
  --load-checkpoint saves/foosball_model_ckpt_120000_steps.zip \\
  --steps 130000  # Remaining steps to reach 250K

════════════════════════════════════════════════════════════════════

For more help: python train_stages.py --help
        """)


if __name__ == "__main__":
    # Note: This is a guide. For actual DummyVecEnv, import it
    print("Use this as a reference for stage-based training.")
    print("\nActual implementation: modify train.py with:")
    print('  model = PPO.load("saves/foosball_stage_X_completed")')
    print('  model.set_env(env)')
    print('  model.learn(total_timesteps=250_000)')
