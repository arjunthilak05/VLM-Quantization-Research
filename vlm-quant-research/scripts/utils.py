"""
Utility functions for VLM quantization research
"""

import os
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: Dict, output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def load_results(results_path: str) -> Dict:
    """Load results from JSON file"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def get_model_size(model) -> float:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def get_gpu_memory() -> Dict[str, float]:
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated
    }


def measure_inference_time(model, inputs, num_iterations: int = 100, warmup: int = 5):
    """Measure average inference time"""
    import time

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(**inputs)

    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(**inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations

    return avg_time * 1000  # Convert to milliseconds


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def log_system_info():
    """Log system and GPU information"""
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
    logger.info("=" * 60)


def prepare_calibration_data(dataset_name: str, num_samples: int = 512, seed: int = 42):
    """Prepare calibration dataset for PTQ methods"""
    from datasets import load_dataset
    import random

    random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Loading calibration dataset: {dataset_name}")

    # Load LLaVA-Instruct dataset or similar
    # For now, return placeholder - implement based on actual dataset
    calibration_samples = []

    try:
        # Example: Load from HuggingFace
        dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")

        # Sample random indices
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

        for idx in indices:
            sample = dataset[idx]
            calibration_samples.append(sample)

        logger.info(f"Prepared {len(calibration_samples)} calibration samples")

    except Exception as e:
        logger.warning(f"Could not load calibration dataset: {e}")
        logger.warning("Using empty calibration set - quantization may be affected")

    return calibration_samples


def format_results_table(results: Dict) -> pd.DataFrame:
    """Format results as pandas DataFrame"""
    rows = []
    for method, metrics in results.items():
        row = {"Method": method}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


class WandbLogger:
    """Wrapper for Weights & Biases logging"""

    def __init__(self, project_name: str, experiment_name: str, config: Dict, enabled: bool = True):
        self.enabled = enabled

        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                self.run = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config
                )
                logger.info(f"W&B logging enabled: {project_name}/{experiment_name}")
            except ImportError:
                logger.warning("wandb not installed, logging disabled")
                self.enabled = False

    def log(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to W&B"""
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def finish(self):
        """Finish W&B run"""
        if self.enabled:
            self.run.finish()


def print_header(text: str, char: str = "="):
    """Print formatted header"""
    width = 80
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_results_summary(results: Dict):
    """Print formatted results summary"""
    print_header("RESULTS SUMMARY")

    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print("-" * 40)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    print("\n" + "=" * 80)
