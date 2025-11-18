"""
Baseline Evaluation Script
Evaluate full-precision models on all benchmarks to establish accuracy baseline
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Dict, List
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils import (
    load_config, save_results, get_model_size,
    measure_inference_time, log_system_info,
    print_header, WandbLogger
)

# Import VLMEvalKit
try:
    from vlmeval.config import supported_VLM
    from vlmeval.utils import track_progress_rich
except ImportError:
    print("VLMEvalKit not installed. Please install: pip install vlmeval")
    sys.exit(1)

from transformers import AutoModelForVision2Seq, AutoProcessor


class BaselineEvaluator:
    """Evaluator for full-precision baseline models"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config['hardware']['device']
        self.precision = config['hardware']['precision']
        self.benchmarks = [b['name'] for b in config['benchmarks'] if b['enabled']]

    def load_model(self, model_config: Dict):
        """Load full-precision model"""
        print_header(f"Loading Model: {model_config['name']}")

        hf_path = model_config['hf_path']

        # Determine dtype
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }
        dtype = dtype_map.get(self.precision, torch.bfloat16)

        print(f"Loading from: {hf_path}")
        print(f"Precision: {self.precision} ({dtype})")

        try:
            # Load model
            model = AutoModelForVision2Seq.from_pretrained(
                hf_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )

            # Load processor
            processor = AutoProcessor.from_pretrained(
                hf_path,
                trust_remote_code=True
            )

            print(f"Model loaded successfully")
            print(f"Model size: {get_model_size(model):.2f} MB")

            return model, processor

        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def evaluate_model_vlmeval(self, model_name: str, hf_path: str) -> Dict:
        """Evaluate using VLMEvalKit CLI interface"""
        import subprocess

        print_header(f"Evaluating {model_name} on Benchmarks")

        results = {}

        for benchmark in self.benchmarks:
            print(f"\n{'='*60}")
            print(f"Benchmark: {benchmark}")
            print(f"{'='*60}\n")

            try:
                # Use VLMEvalKit CLI for evaluation
                cmd = [
                    "python", "-m", "vlmeval.run",
                    "--data", benchmark,
                    "--model", hf_path,
                    "--work-dir", f"experiments/baseline/{model_name}",
                    "--verbose"
                ]

                # Run evaluation
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout per benchmark
                )

                if result.returncode == 0:
                    # Parse results from output
                    # VLMEvalKit typically saves results to JSON
                    result_file = f"experiments/baseline/{model_name}/{benchmark}_result.json"
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            bench_result = json.load(f)
                            results[benchmark] = bench_result.get('score', 0.0)
                    else:
                        print(f"Warning: Result file not found for {benchmark}")
                        results[benchmark] = None
                else:
                    print(f"Error running benchmark {benchmark}: {result.stderr}")
                    results[benchmark] = None

            except Exception as e:
                print(f"Exception during evaluation of {benchmark}: {e}")
                results[benchmark] = None

        return results

    def measure_efficiency_metrics(self, model, processor, model_name: str) -> Dict:
        """Measure efficiency metrics (latency, memory)"""
        print_header(f"Measuring Efficiency Metrics")

        # Prepare sample input
        from PIL import Image
        import requests
        from io import BytesIO

        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (224, 224), color='red')
        dummy_text = "What is in this image?"

        # Process inputs
        inputs = processor(
            images=dummy_image,
            text=dummy_text,
            return_tensors="pt"
        ).to(self.device)

        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Measure inference time
        latency_ms = measure_inference_time(model, inputs, num_iterations=50, warmup=5)

        # Measure memory
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory_mb = 0.0

        # Get model size
        model_size_mb = get_model_size(model)

        metrics = {
            "latency_ms": latency_ms,
            "peak_memory_mb": peak_memory_mb,
            "model_size_mb": model_size_mb,
            "throughput_imgs_per_sec": 1000.0 / latency_ms if latency_ms > 0 else 0.0
        }

        print(f"Latency: {latency_ms:.2f} ms")
        print(f"Peak Memory: {peak_memory_mb:.2f} MB")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print(f"Throughput: {metrics['throughput_imgs_per_sec']:.2f} imgs/sec")

        return metrics

    def run_evaluation(self, model_config: Dict) -> Dict:
        """Run complete evaluation for a model"""
        model_name = model_config['name']
        hf_path = model_config['hf_path']

        print_header(f"BASELINE EVALUATION: {model_name}", char="#")

        # Load model
        model, processor = self.load_model(model_config)

        if model is None:
            print(f"Skipping {model_name} due to loading error")
            return {}

        # Evaluate on benchmarks
        benchmark_results = self.evaluate_model_vlmeval(model_name, hf_path)

        # Measure efficiency metrics
        efficiency_metrics = self.measure_efficiency_metrics(model, processor, model_name)

        # Combine results
        results = {
            "model_name": model_name,
            "hf_path": hf_path,
            "benchmarks": benchmark_results,
            "efficiency": efficiency_metrics,
            "quantization_method": "none (fp16/bf16)"
        }

        # Calculate average accuracy
        valid_scores = [s for s in benchmark_results.values() if s is not None]
        if valid_scores:
            results["avg_accuracy"] = sum(valid_scores) / len(valid_scores)
        else:
            results["avg_accuracy"] = 0.0

        return results


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation of VLMs")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                        help="Path to config file")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific models to evaluate (default: all)")
    parser.add_argument("--output", type=str, default="results/baseline_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    # Log system info
    log_system_info()

    # Load configuration
    config = load_config(args.config)

    # Filter models if specified
    models_to_eval = config['models']
    if args.models:
        models_to_eval = [m for m in models_to_eval if m['name'] in args.models]

    # Sort by priority
    models_to_eval = sorted(models_to_eval, key=lambda x: x.get('priority', 999))

    # Initialize evaluator
    evaluator = BaselineEvaluator(config)

    # Run evaluations
    all_results = {}

    for model_config in models_to_eval:
        model_name = model_config['name']

        try:
            results = evaluator.run_evaluation(model_config)
            all_results[model_name] = results

            # Save intermediate results
            save_results(all_results, args.output)

            print(f"\n✓ Completed evaluation for {model_name}")

        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

        # Clean up to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final results
    save_results(all_results, args.output)

    print_header("BASELINE EVALUATION COMPLETE", char="#")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
