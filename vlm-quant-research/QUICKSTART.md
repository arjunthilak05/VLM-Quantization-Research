# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Environment Setup (5-10 minutes)

```bash
cd vlm-quant-research
chmod +x scripts/00_setup_environment.sh
bash scripts/00_setup_environment.sh
```

### Step 2: Choose Your Path

#### Option A: Quick Test (1-2 hours)
**Test quantization on a single model without full evaluation**

```bash
# Test GPTQ on Qwen2-VL-7B with 4-bit quantization
python scripts/02_ptq_gptq.py --model qwen2_vl_7b --bits 4
```

#### Option B: PTQ Comparison (4-6 hours)
**Compare all PTQ methods without QAT training**

```bash
# Run each method in sequence
python scripts/01_baseline_eval.py --models qwen2_vl_7b
python scripts/02_ptq_gptq.py --model qwen2_vl_7b
python scripts/03_ptq_awq.py --model qwen2_vl_7b
python scripts/04_ptq_bnb.py --model qwen2_vl_7b
python scripts/06_comprehensive_eval.py
python scripts/07_analysis.py
```

#### Option C: Full Pipeline (overnight, 15-20 hours)
**Complete PTQ vs QAT comparison study**

```bash
chmod +x run_full_experiment.sh
bash run_full_experiment.sh qwen2_vl_7b
```

## 📊 Understanding the Output

### After Quantization
```bash
ls models/
# You'll see directories like:
# - qwen2_vl_7b-gptq-int4/
# - qwen2_vl_7b-awq-int4/
# - qwen2_vl_7b-qat-int4/
```

### After Evaluation
```bash
ls results/
# You'll see files like:
# - baseline_results.json
# - gptq_results.json
# - comprehensive_results.json
```

### After Analysis
```bash
ls results/plots/
# - accuracy_vs_size.png
# - accuracy_vs_latency.png
# - accuracy_recovery.png
# - multi_metric_comparison.png

cat results/analysis_report.md
# Markdown report with key findings
```

## 🔧 Common Customizations

### Change Target Model

Edit `configs/experiment_config.yaml`:
```yaml
models:
  - name: "your_model"
    hf_path: "huggingface/model-path"
    priority: 1
```

### Change Quantization Bits

```yaml
quantization:
  ptq_methods:
    gptq:
      bits: [4, 8]  # Test both 4-bit and 8-bit
```

### Add/Remove Benchmarks

```yaml
benchmarks:
  - name: "MMBench_DEV_EN"
    enabled: true  # Set to false to disable
```

## 📈 Quick Results Check

### View Results Summary
```bash
python -c "
import json
with open('results/comprehensive_results.json') as f:
    results = json.load(f)
    for r in results:
        print(f\"{r['method'].upper()}: Accuracy={r.get('avg_accuracy', 0):.2f}%, Size={r.get('model_size_mb', 0):.0f}MB\")
"
```

### Generate Quick Plot
```bash
python scripts/07_analysis.py --results results/comprehensive_results.json
```

## ⚡ Performance Tips

### Speed Up Experiments
1. **Use fewer calibration samples**: Edit `configs/experiment_config.yaml`
   ```yaml
   calibration:
     num_samples: 128  # Reduce from 512
   ```

2. **Test on single benchmark**:
   ```yaml
   benchmarks:
     - name: "MMBench_DEV_EN"
       enabled: true
     - name: "TextVQA_VAL"
       enabled: false  # Disable others
   ```

3. **Skip QAT initially**: QAT takes 6-8 hours, test PTQ methods first

### Optimize Memory Usage
1. **Reduce batch size**:
   ```yaml
   qat:
     batch_size: 2  # Reduce from 4
   ```

2. **Use gradient checkpointing** (already enabled in scripts)

3. **Process models sequentially** (default behavior)

## 🐛 Quick Fixes

### "CUDA out of memory"
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in config
# Then restart the script
```

### "Model not found"
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN=your_token_here
```

### "VLMEvalKit import error"
```bash
pip uninstall vlmeval -y
pip install git+https://github.com/open-compass/VLMEvalKit.git
```

## 📝 Next Steps

After getting results:

1. **Review the analysis report**:
   ```bash
   cat results/analysis_report.md
   ```

2. **Check visualization plots**:
   ```bash
   open results/plots/accuracy_vs_size.png
   ```

3. **Export results for paper**:
   ```bash
   cat results/tables/summary_results.tex  # LaTeX table
   cat results/tables/summary_results.csv  # CSV for Excel
   ```

## 🎓 Learn More

- **Full documentation**: See `README.md`
- **Configuration guide**: See `configs/experiment_config.yaml` comments
- **Method details**: Check individual script docstrings

## 💬 Getting Help

1. Check the troubleshooting section in `README.md`
2. Review script outputs in `logs/` directory
3. Open an issue on GitHub

---

**Happy quantizing! 🚀**
