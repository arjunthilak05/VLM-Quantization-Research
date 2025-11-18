"""
BitsAndBytes (NF4) Post-Training Quantization Script
Uses bitsandbytes library for on-the-fly quantization of VLMs
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Dict
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils import (
    load_config, save_results, get_model_size,
    log_system_info, print_header
)

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


class BnBQuantizer:
    """BitsAndBytes quantization for VLMs"""

    def __init__(self, config: Dict):
        self.config = config
        self.bnb_config = config['quantization']['ptq_methods']['bnb']
        self.device = config['hardware']['device']

    def get_bnb_config(self, bits: int = 4):
        """Create BitsAndBytes quantization config"""

        # Determine compute dtype
        compute_dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }
        compute_dtype = compute_dtype_map.get(
            self.bnb_config['compute_dtype'],
            torch.bfloat16
        )

        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_config['quant_type'],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_config['double_quant'],
            )
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            raise ValueError(f"Unsupported bit width for BnB: {bits}")

        return bnb_config

    def load_quantized_model(self, model_name: str, hf_path: str, bits: int = 4):
        """Load model with BitsAndBytes quantization"""
        print_header(f"BnB Quantization: {model_name} (INT{bits})", char="#")

        print(f"Source model: {hf_path}")
        print(f"Bits: {bits}")
        print(f"Quantization type: {self.bnb_config['quant_type']}")
        print(f"Compute dtype: {self.bnb_config['compute_dtype']}")

        # Get quantization config
        bnb_config = self.get_bnb_config(bits)

        print("\nQuantization Config:")
        print(f"  load_in_4bit: {getattr(bnb_config, 'load_in_4bit', False)}")
        print(f"  load_in_8bit: {getattr(bnb_config, 'load_in_8bit', False)}")
        if bits == 4:
            print(f"  quant_type: {bnb_config.bnb_4bit_quant_type}")
            print(f"  compute_dtype: {bnb_config.bnb_4bit_compute_dtype}")
            print(f"  double_quant: {bnb_config.bnb_4bit_use_double_quant}")

        try:
            # Load model with quantization
            print("\nLoading model with BnB quantization...")
            print("(Note: BnB applies quantization on-the-fly during loading)")

            model = AutoModelForVision2Seq.from_pretrained(
                hf_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )

            # Load processor
            processor = AutoProcessor.from_pretrained(
                hf_path,
                trust_remote_code=True
            )

            print(f"\n✓ BnB quantization complete!")
            print(f"  Model loaded with INT{bits} quantization")

            # Note: We don't save BnB models separately as they're quantized on-the-fly
            # The quantization is applied during loading

            return model, processor, hf_path

        except Exception as e:
            print(f"\n✗ Error during BnB quantization: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def quantize_all_configs(self, model_config: Dict) -> Dict:
        """Test BnB quantization with all configured bit widths"""
        model_name = model_config['name']
        hf_path = model_config['hf_path']

        results = {}

        for bits in self.bnb_config['bits']:
            quant_key = f"bnb_int{bits}"

            print(f"\n{'='*80}")
            print(f"Loading {model_name} with BnB INT{bits}")
            print(f"{'='*80}\n")

            model, processor, model_path = self.load_quantized_model(
                model_name, hf_path, bits
            )

            if model is not None:
                # Get model size (approximated)
                # Note: BnB quantized models show smaller actual memory usage
                try:
                    size_mb = get_model_size(model)
                except:
                    size_mb = None

                results[quant_key] = {
                    "model_path": model_path,  # Original HF path
                    "bits": bits,
                    "method": "bnb",
                    "quant_type": self.bnb_config['quant_type'],
                    "status": "success",
                    "note": "On-the-fly quantization (not saved separately)"
                }

                if size_mb:
                    results[quant_key]["approx_size_mb"] = size_mb

                # Clean up
                del model, processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                results[quant_key] = {
                    "model_path": None,
                    "bits": bits,
                    "method": "bnb",
                    "status": "failed"
                }

        return results


def main():
    parser = argparse.ArgumentParser(description="BitsAndBytes quantization for VLMs")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to quantize (default: all)")
    parser.add_argument("--bits", type=int, default=None,
                        help="Specific bit width (default: all configured)")
    parser.add_argument("--output", type=str, default="results/bnb_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    # Log system info
    log_system_info()

    # Load configuration
    config = load_config(args.config)

    # Override bits if specified
    if args.bits:
        config['quantization']['ptq_methods']['bnb']['bits'] = [args.bits]

    # Filter models if specified
    models_to_quant = config['models']
    if args.model:
        models_to_quant = [m for m in models_to_quant if m['name'] == args.model]

    # Sort by priority
    models_to_quant = sorted(models_to_quant, key=lambda x: x.get('priority', 999))

    # Initialize quantizer
    quantizer = BnBQuantizer(config)

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

            print(f"\n✓ Completed BnB quantization test for {model_name}")

        except Exception as e:
            print(f"\n✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save final results
    save_results(all_results, args.output)

    print_header("BnB QUANTIZATION COMPLETE", char="#")
    print(f"Results saved to: {args.output}")
    print("\nNote: BnB models are quantized on-the-fly and not saved separately.")
    print("To use them, load with the same quantization_config during inference.")


if __name__ == "__main__":
    main()
