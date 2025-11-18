"""
AWQ Post-Training Quantization Script
Uses AutoAWQ library for quantizing VLMs
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
    prepare_calibration_data, log_system_info,
    print_header
)

# Import AutoAWQ
try:
    from awq import AutoAWQForCausalLM
    from awq.quantize.quantizer import AwqQuantizer
except ImportError:
    print("AutoAWQ not installed. Please install: pip install autoawq")
    sys.exit(1)

from transformers import AutoTokenizer
from datasets import load_dataset


class AWQQuantizer:
    """AWQ quantization for VLMs"""

    def __init__(self, config: Dict):
        self.config = config
        self.awq_config = config['quantization']['ptq_methods']['awq']
        self.device = config['hardware']['device']

    def prepare_calibration_dataset(self, tokenizer, num_samples: int = 128):
        """Prepare calibration dataset for AWQ"""
        print_header("Preparing Calibration Dataset")

        calib_config = self.config['calibration']
        dataset_name = calib_config['dataset']

        print(f"Dataset: {dataset_name}")
        print(f"Number of samples: {num_samples}")

        calibration_data = []

        try:
            # Load LLaVA-Instruct dataset
            dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")

            # Sample data
            import random
            random.seed(calib_config['seed'])
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

            for idx in indices:
                sample = dataset[idx]

                # Extract text conversations
                conversations = sample.get('conversations', [])
                if conversations:
                    # Get text prompts for calibration
                    text = " ".join([conv.get('value', '') for conv in conversations])
                    calibration_data.append(text)

            print(f"Prepared {len(calibration_data)} calibration samples")

        except Exception as e:
            print(f"Warning: Could not load calibration dataset: {e}")
            print("Using dummy calibration data")

            # Fallback to dummy data
            calibration_data = [
                "What is in this image? Describe it in detail.",
                "Can you tell me what objects you see?",
                "Explain what is happening in this picture.",
                "What colors do you see in the image?",
                "How many people are visible?",
            ] * (num_samples // 5)

        return calibration_data[:num_samples]

    def quantize_model(self, model_name: str, hf_path: str, bits: int = 4) -> str:
        """Quantize a model using AWQ"""
        print_header(f"AWQ Quantization: {model_name} (INT{bits})", char="#")

        output_dir = f"models/{model_name}-awq-int{bits}"
        os.makedirs(output_dir, exist_ok=True)

        # Check if already quantized
        if os.path.exists(os.path.join(output_dir, "config.json")):
            print(f"Model already quantized at {output_dir}")
            return output_dir

        print(f"Source model: {hf_path}")
        print(f"Output directory: {output_dir}")
        print(f"Bits: {bits}")
        print(f"Group size: {self.awq_config['group_size']}")

        # AWQ quantization config
        quant_config = {
            "zero_point": self.awq_config['zero_point'],
            "q_group_size": self.awq_config['group_size'],
            "w_bit": bits,
            "version": self.awq_config['version']
        }

        print("\nQuantization Config:")
        for key, value in quant_config.items():
            print(f"  {key}: {value}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                hf_path,
                trust_remote_code=True
            )

            # Load model
            print("\nLoading model...")
            model = AutoAWQForCausalLM.from_pretrained(
                hf_path,
                trust_remote_code=True,
                safetensors=True,
                device_map="cpu"  # Load on CPU first
            )

            print(f"Original model size: {get_model_size(model.model):.2f} MB")

            # Prepare calibration data
            calibration_texts = self.prepare_calibration_dataset(
                tokenizer,
                num_samples=128  # AWQ typically uses fewer samples
            )

            # Quantize
            print("\nQuantizing model (this may take 10-30 minutes)...")
            print("Progress will be displayed below:")

            model.quantize(
                tokenizer,
                quant_config=quant_config,
                calib_data=calibration_texts
            )

            # Save quantized model
            print(f"\nSaving quantized model to {output_dir}...")
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)

            print(f"\n✓ AWQ quantization complete!")
            print(f"  Quantized model saved to: {output_dir}")

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output_dir

        except Exception as e:
            print(f"\n✗ Error during AWQ quantization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def quantize_all_configs(self, model_config: Dict) -> Dict:
        """Quantize model with all configured bit widths"""
        model_name = model_config['name']
        hf_path = model_config['hf_path']

        results = {}

        for bits in self.awq_config['bits']:
            quant_key = f"awq_int{bits}"

            print(f"\n{'='*80}")
            print(f"Quantizing {model_name} to INT{bits}")
            print(f"{'='*80}\n")

            output_dir = self.quantize_model(model_name, hf_path, bits)

            if output_dir:
                results[quant_key] = {
                    "model_path": output_dir,
                    "bits": bits,
                    "method": "awq",
                    "status": "success"
                }
            else:
                results[quant_key] = {
                    "model_path": None,
                    "bits": bits,
                    "method": "awq",
                    "status": "failed"
                }

        return results


def main():
    parser = argparse.ArgumentParser(description="AWQ quantization for VLMs")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to quantize (default: all)")
    parser.add_argument("--bits", type=int, default=None,
                        help="Specific bit width (default: all configured)")
    parser.add_argument("--output", type=str, default="results/awq_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    # Log system info
    log_system_info()

    # Load configuration
    config = load_config(args.config)

    # Override bits if specified
    if args.bits:
        config['quantization']['ptq_methods']['awq']['bits'] = [args.bits]

    # Filter models if specified
    models_to_quant = config['models']
    if args.model:
        models_to_quant = [m for m in models_to_quant if m['name'] == args.model]

    # Sort by priority
    models_to_quant = sorted(models_to_quant, key=lambda x: x.get('priority', 999))

    # Initialize quantizer
    quantizer = AWQQuantizer(config)

    # Run quantization
    all_results = {}

    for model_config in models_to_quant:
        model_name = model_config['name']

        try:
            print_header(f"Processing Model: {model_name}", char="#")

            results = quantizer.quantize_all_configs(model_config)
            all_results[model_name] = results

            # Save intermediate results
            save_results(all_results, args.output)

            print(f"\n✓ Completed AWQ quantization for {model_name}")

        except Exception as e:
            print(f"\n✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save final results
    save_results(all_results, args.output)

    print_header("AWQ QUANTIZATION COMPLETE", char="#")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
