#!/usr/bin/env python3
"""
Master test runner for all curriculum stages
"""

import subprocess
import sys


def run_stage_test(stage):
    """Run test for a specific stage"""
    print(f"\n{'='*80}")
    print(f"Running Stage {stage} Tests...")
    print('='*80)
    
    result = subprocess.run(
        [sys.executable, f"test_stage_{stage}.py"],
        cwd=".",
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Run all stage tests"""
    print(f"\n{'='*80}")
    print("FOOSBALL RL CURRICULUM - COMPREHENSIVE TEST SUITE")
    print('='*80)
    print("Running tests for all 4 curriculum stages...")
    
    stages = [1, 2, 3, 4]
    results = {}
    
    for stage in stages:
        try:
            results[stage] = run_stage_test(stage)
        except Exception as e:
            print(f"\n❌ ERROR running Stage {stage}: {e}")
            results[stage] = False
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print('='*80)
    
    for stage in stages:
        status = "✅ PASS" if results[stage] else "❌ FAIL"
        print(f"Stage {stage}: {status}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print(f"\n{'='*80}")
        print("✅ ALL STAGES PASSED - SYSTEM READY FOR TRAINING")
        print('='*80)
        print("\nNext steps:")
        print("  1. uv run train_stages.py --stage 1")
        print("  2. uv run train_stages.py --stage 2 --load saves/foosball_stage_1_completed.zip")
        print("  3. uv run train_stages.py --stage 3 --load saves/foosball_stage_2_completed.zip")
        print("  4. uv run train_stages.py --stage 4 --load saves/foosball_stage_3_completed.zip")
        return 0
    else:
        print(f"\n{'='*80}")
        print("❌ SOME STAGES FAILED")
        print('='*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
