"""
Comprehensive Evaluation Script
Evaluate all quantized models on all benchmarks and measure efficiency metrics
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Dict, List
import json
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils import (
    load_config, save_results, load_results, get_model_size,
    measure_inference_time, get_gpu_memory,
    log_system_info, print_header, format_results_table
)

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig
)
from PIL import Image


class ComprehensiveEvaluator:
    """Comprehensive evaluator for all quantized models"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config['hardware']['device']
        self.benchmarks = [b['name'] for b in config['benchmarks'] if b['enabled']]

    def load_model_by_method(self, model_path: str, method: str, bits: int = 4):
        """Load quantized model based on method"""
        print(f"Loading model: {model_path}")
        print(f"Method: {method}, Bits: {bits}")

        try:
            if method == "bnb":
                # BitsAndBytes - need to apply quantization config
                if bits == 4:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )

                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )

            else:
                # GPTQ, AWQ, QAT - already quantized
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True
                )

            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            print("✓ Model loaded successfully")
            return model, processor

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None, None

    def evaluate_on_benchmarks(self, model_name: str, model_path: str, method: str) -> Dict:
        """Evaluate model on all benchmarks using VLMEvalKit"""
        print_header(f"Benchmarking: {model_name} ({method})")

        results = {}

        for benchmark in self.benchmarks:
            print(f"\nEvaluating on {benchmark}...")

            try:
                # Use VLMEvalKit CLI
                import subprocess

                cmd = [
                    "python", "-m", "vlmeval.run",
                    "--data", benchmark,
                    "--model", model_path,
                    "--work-dir", f"experiments/{method}/{model_name}",
                    "--verbose"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600
                )

                if result.returncode == 0:
                    # Parse results
                    result_file = f"experiments/{method}/{model_name}/{benchmark}_result.json"
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            bench_result = json.load(f)
                            results[benchmark] = bench_result.get('score', 0.0)
                    else:
                        results[benchmark] = None
                else:
                    print(f"  Warning: Benchmark failed - {result.stderr[:200]}")
                    results[benchmark] = None

            except Exception as e:
                print(f"  Error: {e}")
                results[benchmark] = None

        # Calculate average
        valid_scores = [s for s in results.values() if s is not None]
        avg_accuracy = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return {
            "benchmarks": results,
            "avg_accuracy": avg_accuracy
        }

    def measure_efficiency(self, model, processor, model_name: str) -> Dict:
        """Measure efficiency metrics"""
        print_header(f"Measuring Efficiency: {model_name}")

        # Prepare sample input
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        dummy_text = "What is shown in this image?"

        inputs = processor(
            images=dummy_image,
            text=dummy_text,
            return_tensors="pt"
        ).to(self.device)

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Measure latency
        print("Measuring latency...")
        latency_ms = measure_inference_time(model, inputs, num_iterations=100, warmup=10)

        # Measure memory
        print("Measuring memory...")
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory_mb = 0.0

        # Get model size
        try:
            model_size_mb = get_model_size(model)
        except:
            model_size_mb = None

        metrics = {
            "latency_ms": latency_ms,
            "peak_memory_mb": peak_memory_mb,
            "throughput_imgs_per_sec": 1000.0 / latency_ms if latency_ms > 0 else 0.0
        }

        if model_size_mb:
            metrics["model_size_mb"] = model_size_mb

        print(f"  Latency: {latency_ms:.2f} ms")
        print(f"  Peak Memory: {peak_memory_mb:.2f} MB")
        print(f"  Throughput: {metrics['throughput_imgs_per_sec']:.2f} imgs/sec")
        if model_size_mb:
            print(f"  Model Size: {model_size_mb:.2f} MB")

        return metrics

    def evaluate_model(self, model_info: Dict, baseline_results: Dict = None) -> Dict:
        """Evaluate a single quantized model"""
        model_name = model_info['model_name']
        model_path = model_info['model_path']
        method = model_info['method']
        bits = model_info.get('bits', 4)

        print_header(f"EVALUATING: {model_name} - {method.upper()} INT{bits}", char="#")

        # Load model
        model, processor = self.load_model_by_method(model_path, method, bits)

        if model is None:
            return {
                "model_name": model_name,
                "method": method,
                "bits": bits,
                "status": "failed",
                "error": "Could not load model"
            }

        # Evaluate on benchmarks
        benchmark_results = self.evaluate_on_benchmarks(model_name, model_path, method)

        # Measure efficiency
        efficiency_metrics = self.measure_efficiency(model, processor, model_name)

        # Combine results
        results = {
            "model_name": model_name,
            "method": method,
            "bits": bits,
            "model_path": model_path,
            **benchmark_results,
            **efficiency_metrics,
            "status": "success"
        }

        # Calculate metrics relative to baseline
        if baseline_results and model_name in baseline_results:
            baseline = baseline_results[model_name]

            if "avg_accuracy" in baseline:
                baseline_acc = baseline["avg_accuracy"]
                results["accuracy_recovery_%"] = (results["avg_accuracy"] / baseline_acc * 100) if baseline_acc > 0 else 0

            if "efficiency" in baseline and "latency_ms" in baseline["efficiency"]:
                baseline_latency = baseline["efficiency"]["latency_ms"]
                results["speedup"] = baseline_latency / efficiency_metrics["latency_ms"] if efficiency_metrics["latency_ms"] > 0 else 0

            if "efficiency" in baseline and "model_size_mb" in baseline["efficiency"]:
                baseline_size = baseline["efficiency"]["model_size_mb"]
                if "model_size_mb" in efficiency_metrics:
                    results["compression_ratio"] = baseline_size / efficiency_metrics["model_size_mb"] if efficiency_metrics["model_size_mb"] > 0 else 0

        # Clean up
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results


def collect_quantized_models(config: Dict) -> List[Dict]:
    """Collect all quantized models from previous steps"""
    print_header("Collecting Quantized Models")

    models_to_eval = []

    # Load results from previous quantization steps
    result_files = {
        "gptq": "results/gptq_results.json",
        "awq": "results/awq_results.json",
        "bnb": "results/bnb_results.json",
        "qat": "results/qat_results.json"
    }

    for method, result_file in result_files.items():
        if os.path.exists(result_file):
            print(f"\nLoading {method.upper()} results from {result_file}")

            try:
                results = load_results(result_file)

                for model_name, model_results in results.items():
                    if isinstance(model_results, dict):
                        for quant_config, info in model_results.items():
                            if isinstance(info, dict) and info.get('status') == 'success':
                                model_path = info.get('model_path')
                                bits = info.get('bits', 4)

                                if model_path:
                                    models_to_eval.append({
                                        'model_name': model_name,
                                        'model_path': model_path,
                                        'method': method,
                                        'bits': bits,
                                        'quant_config': quant_config
                                    })

                print(f"  Found {len([m for m in models_to_eval if m['method'] == method])} {method} models")

            except Exception as e:
                print(f"  Warning: Could not load {result_file}: {e}")

    print(f"\nTotal models to evaluate: {len(models_to_eval)}")

    return models_to_eval


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of all quantized models")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                        help="Path to config file")
    parser.add_argument("--baseline", type=str, default="results/baseline_results.json",
                        help="Path to baseline results")
    parser.add_argument("--output", type=str, default="results/comprehensive_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    # Log system info
    log_system_info()

    # Load configuration
    config = load_config(args.config)

    # Load baseline results
    baseline_results = None
    if os.path.exists(args.baseline):
        print(f"Loading baseline results from {args.baseline}")
        baseline_results = load_results(args.baseline)

    # Collect all quantized models
    models_to_eval = collect_quantized_models(config)

    if not models_to_eval:
        print("\n✗ No quantized models found to evaluate!")
        print("Please run quantization scripts first (02_ptq_gptq.py, etc.)")
        return

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(config)

    # Evaluate all models
    all_results = []

    for model_info in models_to_eval:
        try:
            results = evaluator.evaluate_model(model_info, baseline_results)
            all_results.append(results)

            # Save intermediate results
            save_results(all_results, args.output)

            print(f"\n✓ Completed evaluation")

        except Exception as e:
            print(f"\n✗ Error evaluating model: {e}")
            import traceback
            traceback.print_exc()

    # Save final results
    save_results(all_results, args.output)

    # Print summary table
    print_header("EVALUATION COMPLETE", char="#")
    print(f"Results saved to: {args.output}")

    # Create summary table
    df = format_results_table({r['model_name'] + "_" + r['method']: r for r in all_results})
    print("\n" + df.to_string())


if __name__ == "__main__":
    main()
