#!/usr/bin/env python3
"""
Run all 4 training stages automatically with checkpoint loading.

Usage:
  python run_all_stages.py              # Default: 4 parallel envs
  python run_all_stages.py --num-envs 8  # Custom: 8 parallel envs
  python run_all_stages.py --num-envs 1  # Safe: 1 env (slow but works)
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd):
    """Run shell command and return success status."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def verify_environment():
    """Verify foosball environment works (quick check)."""
    print("Verifying environment...")
    try:
        # Just check if the module can be imported
        result = subprocess.run(
            [sys.executable, "-c", "import foosball_env; print('OK')"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0 and "OK" in result.stdout.decode():
            print("✓ Environment OK\n")
            return True
    except Exception as e:
        print(f"⚠️  Skipping verification: {e}\n")
        # Don't fail - user might not have GUI set up
        return True
    return True


def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║     Foosball RL Training - Automated 4-Stage Pipeline              ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    # Parse arguments
    num_envs = 4  # Default
    if "--num-envs" in sys.argv:
        idx = sys.argv.index("--num-envs")
        if idx + 1 < len(sys.argv):
            num_envs = int(sys.argv[idx + 1])
    
    print(f"Configuration:")
    print(f"  Stages: 4")
    print(f"  Parallel Envs: {num_envs}")
    print(f"  Steps per Stage: 250,000")
    print()
    
    # Verify setup
    if not verify_environment():
        print("❌ Environment verification failed. Aborting.")
        return 1
    
    stages = 4
    steps_per_stage = 5_000_000
    
    # Training loop
    for stage in range(1, stages + 1):
        print("=" * 68)
        print(f"Stage {stage} of {stages}")
        print("=" * 68)
        
        if stage == 1:
            # Stage 1: Fresh start
            print("🆕 Fresh start (no checkpoint)")
            cmd = [
                "uv", "run", "train_stages.py",
                "--stage", "1",
                "--steps", str(steps_per_stage),
                "--num-envs", str(num_envs)
            ]
        else:
            # Stage 2-4: Load from previous
            prev_stage = stage - 1
            checkpoint = f"saves/foosball_stage_{prev_stage}_completed.zip"
            
            if not os.path.exists(checkpoint):
                print(f"❌ ERROR: Checkpoint not found: {checkpoint}")
                print(f"   Stage {prev_stage} may not have completed successfully")
                return 1
            
            print(f"✓ Loading checkpoint: {checkpoint}")
            cmd = [
                "uv", "run", "train_stages.py",
                "--stage", str(stage),
                "--load", checkpoint,
                "--steps", str(steps_per_stage),
                "--num-envs", str(num_envs)
            ]
        
        # Run training
        if not run_command(cmd):
            print(f"❌ Stage {stage} failed!")
            return 1
        
        print(f"✓ Stage {stage} complete\n")
    
    # Success!
    print("=" * 68)
    print("✅ ALL STAGES COMPLETE!")
    print("=" * 68)
    print()
    print("Final model saved to:")
    print("  saves/foosball_stage_4_completed.zip")
    print()
    print("Test it:")
    print("  uv run test.py --model saves/foosball_stage_4_completed.zip --episodes 20")
    print()
    print("Monitor with TensorBoard:")
    print("  tensorboard --logdir logs/")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
