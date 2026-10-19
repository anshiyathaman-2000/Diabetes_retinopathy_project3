#!/usr/bin/env bash
set -e

# Run Proposed CNN-Transformer-BiLSTM Training, Evaluation, and Explainability on IDRiD
export PYTHONPATH=.

echo "=========================================================="
echo " Starting Proposed CNN-Transformer-BiLSTM Pipeline"
echo "=========================================================="

# 1. Train Proposed Hybrid Model
python3 src/train.py \
    --config configs/experiment.yaml \
    --model hybrid \
    --epochs 20 \
    --batch_size 16 \
    --lr 0.0001 \
    --preprocessing clahe \
    --seed 42

CHECKPOINT="checkpoints/hybrid_five_class_clahe_s42_best.pt"

# 2. Evaluate on Untouched Test Set
if [ -f "$CHECKPOINT" ]; then
    echo "Evaluating hybrid model on test set..."
    python3 src/evaluate.py --checkpoint "$CHECKPOINT"
    
    # 3. Generate Grad-CAM Visualizations for sample test cases
    echo "Generating Grad-CAM explanations..."
    python3 src/explainability.py \
        --image "B. Disease Grading/1. Original Images/b. Testing Set/IDRiD_001.jpg" \
        --checkpoint "$CHECKPOINT" \
        --gt 4 \
        --save "outputs/explainability/hybrid_test_IDRiD_001_pdr.png"

    python3 src/explainability.py \
        --image "B. Disease Grading/1. Original Images/b. Testing Set/IDRiD_029.jpg" \
        --checkpoint "$CHECKPOINT" \
        --gt 0 \
        --save "outputs/explainability/hybrid_test_IDRiD_029_nodr.png"
else
    echo "Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "Proposed CNN-Transformer-BiLSTM pipeline complete!"
