#!/bin/bash

# Full Experimental Pipeline
# This script runs the complete PTQ vs QAT comparison study

set -e  # Exit on error

echo "======================================================================"
echo "VLM Quantization Research - Full Experimental Pipeline"
echo "======================================================================"

# Configuration
MODEL="${1:-qwen2_vl_7b}"  # Default to Qwen2-VL-7B
CONFIG="configs/experiment_config.yaml"

echo "Model: $MODEL"
echo "Config: $CONFIG"
echo ""

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Function to run and log a script
run_script() {
    local script=$1
    local log_file=$2
    local description=$3

    echo "======================================================================"
    echo "$description"
    echo "======================================================================"
    echo "Script: $script"
    echo "Log: $log_file"
    echo ""

    python "scripts/$script" --config "$CONFIG" --model "$MODEL" 2>&1 | tee "$LOG_DIR/$log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ $description completed successfully"
    else
        echo "✗ $description failed"
        exit 1
    fi

    echo ""
}

# Step 1: Baseline Evaluation
run_script \
    "01_baseline_eval.py" \
    "01_baseline.log" \
    "Step 1: Baseline Evaluation"

# Step 2: PTQ - GPTQ
run_script \
    "02_ptq_gptq.py" \
    "02_gptq.log" \
    "Step 2: GPTQ Quantization"

# Step 3: PTQ - AWQ
run_script \
    "03_ptq_awq.py" \
    "03_awq.log" \
    "Step 3: AWQ Quantization"

# Step 4: PTQ - BitsAndBytes
run_script \
    "04_ptq_bnb.py" \
    "04_bnb.log" \
    "Step 4: BitsAndBytes Quantization"

# Step 5: QAT Training (most time-consuming)
echo "======================================================================"
echo "Step 5: QAT Training (This will take several hours)"
echo "======================================================================"
echo ""

read -p "Do you want to run QAT training? This can take 6-8 hours. (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_script \
        "05_qat_training.py" \
        "05_qat.log" \
        "Step 5: QAT Training"
else
    echo "Skipping QAT training. You can run it later with:"
    echo "  python scripts/05_qat_training.py --config $CONFIG --model $MODEL"
    echo ""
fi

# Step 6: Comprehensive Evaluation
run_script \
    "06_comprehensive_eval.py" \
    "06_eval.log" \
    "Step 6: Comprehensive Evaluation"

# Step 7: Analysis and Visualization
echo "======================================================================"
echo "Step 7: Analysis and Visualization"
echo "======================================================================"
echo ""

python scripts/07_analysis.py \
    --results results/comprehensive_results.json \
    --baseline results/baseline_results.json \
    --output results \
    2>&1 | tee "$LOG_DIR/07_analysis.log"

echo ""
echo "======================================================================"
echo "EXPERIMENTAL PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Results:"
echo "  - Baseline: results/baseline_results.json"
echo "  - GPTQ: results/gptq_results.json"
echo "  - AWQ: results/awq_results.json"
echo "  - BnB: results/bnb_results.json"
echo "  - QAT: results/qat_results.json"
echo "  - Comprehensive: results/comprehensive_results.json"
echo ""
echo "Analysis:"
echo "  - Plots: results/plots/"
echo "  - Tables: results/tables/"
echo "  - Report: results/analysis_report.md"
echo ""
echo "Logs saved to: $LOG_DIR"
echo ""
echo "======================================================================"
