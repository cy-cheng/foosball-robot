#!/usr/bin/env python3
"""
Quick start script to verify environment and run a short training test.
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run command and report status"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ {description} failed!")
        return False
    print(f"✅ {description} succeeded!")
    return True


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       🎮 Foosball RL Training - Quick Start 🎮           ║
    ╚══════════════════════════════════════════════════════════╝
    
    This script will:
    1. ✅ Verify environment setup
    2. ✅ Test basic training (10K steps)
    3. ✅ Show usage examples
    """)
    
    input("Press Enter to continue...\n")
    
    # Step 1: Verify environment
    success = run_command(
        [sys.executable, "foosball_env.py"],
        "Environment verification"
    )
    
    if not success:
        print("\n❌ Environment verification failed. Check dependencies.")
        return
    
    # Step 2: Quick training test
    success = run_command(
        [sys.executable, "train.py", "--mode", "train", "--steps", "10000", "--level", "1", "--no-render"],
        "Quick training test (10K steps)"
    )
    
    if not success:
        print("\n❌ Training test failed.")
        return
    
    # Step 3: Show next steps
    print(f"""
    
    ╔══════════════════════════════════════════════════════════╗
    ║                    ✅ Setup Complete! ✅                 ║
    ╚══════════════════════════════════════════════════════════╝
    
    📖 Next Steps:
    
    1️⃣  Train a full model (1M steps):
        uv run train.py --steps 1000000 --level 1 --num-envs 4
    
    2️⃣  Test the trained model:
        uv run test.py --model saves/foosball_model_final.zip --episodes 5
    
    3️⃣  Monitor training with TensorBoard:
        tensorboard --logdir logs/
    
    4️⃣  Advanced options:
        uv run train.py --help
        uv run test.py --help
    
    📚 For detailed documentation, see TRAINING_README.md
    
    Good luck! 🚀
    """)


if __name__ == "__main__":
    main()
