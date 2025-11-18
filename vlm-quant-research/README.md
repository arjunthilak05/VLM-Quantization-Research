# VLM Quantization Research: PTQ vs QAT Comprehensive Benchmarking

> **A complete research framework for comparing Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) methods for Vision-Language Models**

## 🎯 Research Objectives

This project provides a comprehensive comparison of quantization methods for Vision-Language Models (VLMs), answering key research questions:

1. **How do PTQ and QAT compare in accuracy-efficiency trade-offs?**
2. **What is the accuracy degradation at different quantization levels (INT8, INT4)?**
3. **What are the training cost differences (PTQ ~minutes vs QAT ~hours)?**
4. **How much memory footprint reduction can be achieved?**
5. **What inference speed improvements are possible?**
6. **Which models are more sensitive to quantization?**

## 📊 Quantization Methods Evaluated

### Post-Training Quantization (PTQ)
- **GPTQ**: Generative Pre-trained Transformer Quantization
- **AWQ**: Activation-aware Weight Quantization
- **BitsAndBytes**: NF4 4-bit quantization

### Quantization-Aware Training (QAT)
- **TorchAO QAT**: PyTorch's official QAT implementation

## 🔬 Models Evaluated

1. **Qwen2-VL-7B** - State-of-the-art VLM (Priority 1)
2. **LLaVA-v1.5-7B** - Popular baseline VLM (Priority 2)
3. **MiniCPM-V-2.6** - Efficient edge-deployment model (Priority 3)

## 📈 Benchmarks Used

- **MMBench** - General multimodal understanding
- **TextVQA** - OCR and text understanding
- **ScienceQA** - Reasoning capabilities
- **GQA** - Visual reasoning
- **MMMU** - Expert-level understanding

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Make setup script executable
chmod +x scripts/00_setup_environment.sh

# Run setup (installs all dependencies)
bash scripts/00_setup_environment.sh
```

This will install:
- PyTorch with CUDA 12.1 support
- Transformers, Accelerate
- GPTQModel, AutoAWQ, BitsAndBytes, TorchAO
- VLMEvalKit
- All utility libraries

### 2. Run Quick Experiment (PTQ only, faster)

```bash
# Run PTQ methods only (GPTQ, AWQ, BnB)
# This takes ~3-5 hours on H200 GPU

# Step 1: Baseline
python scripts/01_baseline_eval.py

# Step 2-4: PTQ methods
python scripts/02_ptq_gptq.py
python scripts/03_ptq_awq.py
python scripts/04_ptq_bnb.py

# Step 5: Evaluate
python scripts/06_comprehensive_eval.py

# Step 6: Analyze
python scripts/07_analysis.py
```

### 3. Run Full Pipeline (PTQ + QAT, overnight)

```bash
# Make executable
chmod +x run_full_experiment.sh

# Run complete pipeline (includes QAT training)
# This takes ~15-20 hours on H200 GPU
bash run_full_experiment.sh qwen2_vl_7b
```

## 📂 Project Structure

```
vlm-quant-research/
├── configs/
│   └── experiment_config.yaml      # Experiment configuration
├── scripts/
│   ├── 00_setup_environment.sh     # Environment setup
│   ├── 01_baseline_eval.py         # Baseline evaluation
│   ├── 02_ptq_gptq.py             # GPTQ quantization
│   ├── 03_ptq_awq.py              # AWQ quantization
│   ├── 04_ptq_bnb.py              # BitsAndBytes quantization
│   ├── 05_qat_training.py         # QAT training pipeline
│   ├── 06_comprehensive_eval.py    # Comprehensive evaluation
│   ├── 07_analysis.py             # Analysis & visualization
│   └── utils.py                   # Utility functions
├── results/
│   ├── baseline_results.json
│   ├── gptq_results.json
│   ├── awq_results.json
│   ├── bnb_results.json
│   ├── qat_results.json
│   ├── comprehensive_results.json
│   ├── plots/                     # Generated visualizations
│   ├── tables/                    # Result tables
│   └── analysis_report.md         # Analysis report
├── models/                        # Quantized models
├── experiments/                   # Experiment outputs
├── run_full_experiment.sh         # Main execution script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

Edit `configs/experiment_config.yaml` to customize:

- **Models to evaluate**: Add/remove models
- **Quantization settings**: Bit widths, group sizes, etc.
- **Benchmarks**: Enable/disable specific benchmarks
- **Training parameters**: Learning rate, epochs, batch size
- **Hardware settings**: Device, memory limits

Example:
```yaml
models:
  - name: "qwen2_vl_7b"
    hf_path: "Qwen/Qwen2-VL-7B-Instruct"
    priority: 1

quantization:
  ptq_methods:
    gptq:
      bits: [4, 8]
      group_size: 128
  qat:
    bits: 4
    learning_rate: 2.0e-5
    training_epochs: 3
```

## 📊 Expected Results

After running the full pipeline, you'll get:

### 1. Quantitative Results
- **Accuracy scores** on 5+ benchmarks
- **Model size reduction** (compression ratios)
- **Inference latency** measurements
- **Memory footprint** analysis
- **Accuracy recovery** percentages

### 2. Visualizations
- Accuracy vs Model Size scatter plots
- Accuracy vs Latency plots
- Accuracy recovery bar charts
- Multi-metric comparison charts

### 3. Tables
- CSV summary table
- LaTeX-formatted tables for papers
- Markdown report with key findings

## ⏱️ Time Estimates (on H200 GPU)

| Phase | Time | Description |
|-------|------|-------------|
| Setup | 30 min | Environment & dependencies |
| Baseline Eval | 1-2 h | Full-precision benchmarks |
| GPTQ | 1 h | 10-30 min per model |
| AWQ | 1 h | 10-30 min per model |
| BnB | 30 min | Instant quantization |
| **QAT Training** | **6-8 h** | Most time-consuming |
| Comprehensive Eval | 3-4 h | All models × benchmarks |
| Analysis | 30 min | Plots & tables |
| **Total** | **~15-20 h** | Can run overnight |

## 💡 Usage Tips

### Running Individual Methods

```bash
# Test only GPTQ on a specific model
python scripts/02_ptq_gptq.py --model qwen2_vl_7b --bits 4

# Test only 4-bit quantization across all methods
python scripts/02_ptq_gptq.py --bits 4
python scripts/03_ptq_awq.py --bits 4
python scripts/04_ptq_bnb.py --bits 4
```

### Evaluating Specific Models

```bash
# Evaluate only baseline models
python scripts/01_baseline_eval.py --models qwen2_vl_7b llava_v15_7b

# Skip QAT in full pipeline
# (Edit run_full_experiment.sh and comment out Step 5)
```

### Analyzing Results

```bash
# Generate analysis from existing results
python scripts/07_analysis.py \
    --results results/comprehensive_results.json \
    --baseline results/baseline_results.json \
    --output results
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `configs/experiment_config.yaml`
- Use gradient checkpointing
- Process models sequentially instead of in parallel

### Dataset Loading Issues
```bash
# Pre-download datasets
huggingface-cli login
huggingface-cli download liuhaotian/LLaVA-Instruct-150K
```

### VLMEvalKit Errors
```bash
# Reinstall VLMEvalKit
pip uninstall vlmeval -y
pip install git+https://github.com/open-compass/VLMEvalKit.git
```

## 📚 Key References

- **GPTQModel**: https://github.com/ModelCloud/GPTQModel
- **AutoAWQ**: https://github.com/casper-hansen/AutoAWQ
- **TorchAO**: https://github.com/pytorch/ao
- **VLMEvalKit**: https://github.com/open-compass/VLMEvalKit
- **Qwen2-VL**: https://github.com/QwenLM/Qwen2-VL

## 📝 Citation

If you use this research framework, please cite:

```bibtex
@misc{vlm-quantization-research,
  title={PTQ vs QAT for Vision-Language Models: A Comprehensive Benchmarking Study},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/vlm-quant-research}}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- PyTorch team for TorchAO
- HuggingFace for Transformers
- ModelCloud for GPTQModel
- OpenCompass for VLMEvalKit
- Qwen team for Qwen2-VL

---

**Ready to start your quantization research? Run the setup script and dive in!**

```bash
bash scripts/00_setup_environment.sh
bash run_full_experiment.sh
```
