"""
GPTQ Post-Training Quantization Script
Uses GPTQModel library for quantizing VLMs
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

# Import GPTQModel
try:
    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.quantization import QUANT_CONFIG_MAPPING
except ImportError:
    print("GPTQModel not installed. Please install: pip install gptqmodel")
    sys.exit(1)

from transformers import AutoProcessor, AutoTokenizer
from datasets import load_dataset


class GPTQQuantizer:
    """GPTQ quantization for VLMs"""

    def __init__(self, config: Dict):
        self.config = config
        self.gptq_config = config['quantization']['ptq_methods']['gptq']
        self.device = config['hardware']['device']

    def prepare_calibration_dataset(self, num_samples: int = 512):
        """Prepare calibration dataset for GPTQ"""
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
            ] * (num_samples // 3)

        return calibration_data[:num_samples]

    def quantize_model(self, model_name: str, hf_path: str, bits: int = 4) -> str:
        """Quantize a model using GPTQ"""
        print_header(f"GPTQ Quantization: {model_name} (INT{bits})", char="#")

        output_dir = f"models/{model_name}-gptq-int{bits}"
        os.makedirs(output_dir, exist_ok=True)

        # Check if already quantized
        if os.path.exists(os.path.join(output_dir, "config.json")):
            print(f"Model already quantized at {output_dir}")
            return output_dir

        print(f"Source model: {hf_path}")
        print(f"Output directory: {output_dir}")
        print(f"Bits: {bits}")
        print(f"Group size: {self.gptq_config['group_size']}")

        # Prepare quantization config
        quantize_config = QuantizeConfig(
            bits=bits,
            group_size=self.gptq_config['group_size'],
            desc_act=self.gptq_config['desc_act'],
            sym=self.gptq_config['sym'],
            damp_percent=0.01,
            model_file_base_name="model"
        )

        print("\nQuantization Config:")
        print(f"  bits: {quantize_config.bits}")
        print(f"  group_size: {quantize_config.group_size}")
        print(f"  desc_act: {quantize_config.desc_act}")
        print(f"  sym: {quantize_config.sym}")

        try:
            # Load model
            print("\nLoading model...")
            model = GPTQModel.from_pretrained(
                hf_path,
                quantize_config=quantize_config,
                trust_remote_code=True
            )

            print(f"Original model size: {get_model_size(model):.2f} MB")

            # Prepare calibration data
            calibration_data = self.prepare_calibration_dataset(
                num_samples=self.config['calibration']['num_samples']
            )

            # Load tokenizer for text preparation
            tokenizer = AutoTokenizer.from_pretrained(
                hf_path,
                trust_remote_code=True
            )

            # Tokenize calibration data
            print("\nTokenizing calibration data...")
            calibration_inputs = []
            for text in calibration_data:
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding="max_length"
                )
                calibration_inputs.append(tokens['input_ids'])

            # Quantize
            print("\nQuantizing model (this may take 10-30 minutes)...")
            print("Progress will be displayed below:")

            model.quantize(
                calibration_inputs,
                batch_size=1
            )

            # Save quantized model
            print(f"\nSaving quantized model to {output_dir}...")
            model.save_quantized(output_dir)

            # Save tokenizer/processor
            tokenizer.save_pretrained(output_dir)

            print(f"\n✓ GPTQ quantization complete!")
            print(f"  Quantized model saved to: {output_dir}")

            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output_dir

        except Exception as e:
            print(f"\n✗ Error during GPTQ quantization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def quantize_all_configs(self, model_config: Dict) -> Dict:
        """Quantize model with all configured bit widths"""
        model_name = model_config['name']
        hf_path = model_config['hf_path']

        results = {}

        for bits in self.gptq_config['bits']:
            quant_key = f"gptq_int{bits}"

            print(f"\n{'='*80}")
            print(f"Quantizing {model_name} to INT{bits}")
            print(f"{'='*80}\n")

            output_dir = self.quantize_model(model_name, hf_path, bits)

            if output_dir:
                results[quant_key] = {
                    "model_path": output_dir,
                    "bits": bits,
                    "method": "gptq",
                    "status": "success"
                }
            else:
                results[quant_key] = {
                    "model_path": None,
                    "bits": bits,
                    "method": "gptq",
                    "status": "failed"
                }

        return results


def main():
    parser = argparse.ArgumentParser(description="GPTQ quantization for VLMs")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to quantize (default: all)")
    parser.add_argument("--bits", type=int, default=None,
                        help="Specific bit width (default: all configured)")
    parser.add_argument("--output", type=str, default="results/gptq_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    # Log system info
    log_system_info()

    # Load configuration
    config = load_config(args.config)

    # Override bits if specified
    if args.bits:
        config['quantization']['ptq_methods']['gptq']['bits'] = [args.bits]

    # Filter models if specified
    models_to_quant = config['models']
    if args.model:
        models_to_quant = [m for m in models_to_quant if m['name'] == args.model]

    # Sort by priority
    models_to_quant = sorted(models_to_quant, key=lambda x: x.get('priority', 999))

    # Initialize quantizer
    quantizer = GPTQQuantizer(config)

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

            print(f"\n✓ Completed GPTQ quantization for {model_name}")

        except Exception as e:
            print(f"\n✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save final results
    save_results(all_results, args.output)

    print_header("GPTQ QUANTIZATION COMPLETE", char="#")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
