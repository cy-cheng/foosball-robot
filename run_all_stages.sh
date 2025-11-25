#!/bin/bash

# Complete 4-stage training script
# Trains all curriculum stages sequentially with checkpoint loading

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║          Foosball RL Training - All 4 Stages                      ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo

# Configuration
STAGES=4
NUM_ENVS=${1:-4}  # Default 4, can pass as argument
STEPS=250000

echo "Configuration:"
echo "  Stages: $STAGES"
echo "  Parallel Envs: $NUM_ENVS"
echo "  Steps per Stage: $STEPS"
echo

# Verify setup (quick check)
echo "Verifying environment..."
python -c "import foosball_env; print('OK')" > /dev/null 2>&1 && echo "✓ Environment OK" || echo "⚠️  Skipping verification"
echo

# Training loop
for stage in $(seq 1 $STAGES); do
  echo "════════════════════════════════════════════════════════════════════"
  echo "Stage $stage of $STAGES"
  echo "════════════════════════════════════════════════════════════════════"
  
  if [ $stage -eq 1 ]; then
    # Stage 1: Fresh start
    echo "🆕 Fresh start (no checkpoint)"
    uv run train_stages.py --stage $stage --steps $STEPS --num-envs $NUM_ENVS
  else
    # Stage 2-4: Load from previous
    prev_stage=$((stage - 1))
    checkpoint="saves/foosball_stage_${prev_stage}_completed.zip"
    
    if [ ! -f "$checkpoint" ]; then
      echo "✗ ERROR: Checkpoint not found: $checkpoint"
      echo "  Stage $prev_stage may not have completed successfully"
      exit 1
    fi
    
    echo "✓ Loading checkpoint: $checkpoint"
    uv run train_stages.py --stage $stage --load "$checkpoint" --steps $STEPS --num-envs $NUM_ENVS
  fi
  
  if [ $? -ne 0 ]; then
    echo "✗ Stage $stage failed!"
    exit 1
  fi
  
  echo "✓ Stage $stage complete"
  echo
done

echo "════════════════════════════════════════════════════════════════════"
echo "✅ ALL STAGES COMPLETE!"
echo "════════════════════════════════════════════════════════════════════"
echo
echo "Final model saved to:"
echo "  saves/foosball_stage_4_completed.zip"
echo
echo "Test it:"
echo "  uv run test.py --model saves/foosball_stage_4_completed.zip --episodes 20"
echo
