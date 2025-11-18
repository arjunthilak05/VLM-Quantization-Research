"""
Quantization-Aware Training (QAT) Script
Uses TorchAO for QAT of VLMs
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Dict
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils import (
    load_config, save_results, get_model_size,
    log_system_info, print_header, WandbLogger
)

# Import TorchAO
try:
    from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig, Int4WeightOnlyConfig
    from torchao.quantization.qat import QATConfig
except ImportError:
    print("TorchAO not installed. Please install from source:")
    print("  pip install git+https://github.com/pytorch/ao.git")
    sys.exit(1)

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from datasets import load_dataset
from PIL import Image
import io


class VLMDataset(Dataset):
    """Dataset for VLM training"""

    def __init__(self, hf_dataset, processor, max_samples=None):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_samples = max_samples

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(hf_dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Get image
        image = sample.get('image', None)
        if image is None:
            # Create dummy image if no image present
            image = Image.new('RGB', (224, 224), color='red')

        # Get text
        conversations = sample.get('conversations', [])
        if conversations:
            # Extract question and answer
            question = ""
            answer = ""
            for conv in conversations:
                if conv.get('from') == 'human':
                    question = conv.get('value', '')
                elif conv.get('from') == 'gpt':
                    answer = conv.get('value', '')

            # Combine as prompt
            prompt = f"Question: {question}\nAnswer: {answer}"
        else:
            prompt = sample.get('text', 'Describe this image.')

        return {
            'image': image,
            'text': prompt
        }


class QATTrainer:
    """Quantization-Aware Training trainer"""

    def __init__(self, config: Dict):
        self.config = config
        self.qat_config = config['quantization']['qat']
        self.device = config['hardware']['device']

    def prepare_qat_model(self, model_name: str, hf_path: str):
        """Prepare model for QAT"""
        print_header(f"Preparing Model for QAT: {model_name}")

        print(f"Source model: {hf_path}")
        print(f"Target bits: {self.qat_config['bits']}")
        print(f"Group size: {self.qat_config['group_size']}")

        # Load base model
        print("\nLoading base model...")
        model = AutoModelForVision2Seq.from_pretrained(
            hf_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        processor = AutoProcessor.from_pretrained(
            hf_path,
            trust_remote_code=True
        )

        print(f"Original model size: {get_model_size(model):.2f} MB")

        # Configure QAT
        print("\nConfiguring QAT...")

        # Choose quantization configuration based on settings
        if self.qat_config['activation_bits'] == 8:
            # Int8 activations + Int4 weights
            base_config = Int8DynamicActivationInt4WeightConfig(
                group_size=self.qat_config['group_size']
            )
        else:
            # Int4 weight-only
            base_config = Int4WeightOnlyConfig(
                group_size=self.qat_config['group_size']
            )

        # Create QAT config for prepare step
        qat_config = QATConfig(base_config, step="prepare")

        print(f"QAT Config: {qat_config}")

        # Insert fake quantization operations
        print("Inserting fake quantization operations...")
        quantize_(model, qat_config)

        print("✓ Model prepared for QAT")

        return model, processor, base_config

    def create_dataloader(self, processor, batch_size: int = 4):
        """Create training dataloader"""
        print_header("Preparing Training Data")

        train_config = self.config['training']
        dataset_name = train_config['dataset']
        num_samples = train_config['num_samples']

        print(f"Dataset: {dataset_name}")
        print(f"Number of samples: {num_samples}")
        print(f"Batch size: {batch_size}")

        try:
            # Load LLaVA-Instruct dataset
            hf_dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")

            # Create dataset
            dataset = VLMDataset(hf_dataset, processor, max_samples=num_samples)

            # Create dataloader
            def collate_fn(batch):
                images = [item['image'] for item in batch]
                texts = [item['text'] for item in batch]

                # Process batch
                inputs = processor(
                    images=images,
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                return inputs

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=collate_fn
            )

            print(f"✓ Created dataloader with {len(dataset)} samples")
            print(f"  Batches per epoch: {len(dataloader)}")

            return dataloader

        except Exception as e:
            print(f"Warning: Could not load training dataset: {e}")
            print("QAT training may fail without proper data")
            return None

    def train_qat(self, model, dataloader, epochs: int = 3):
        """Train model with fake quantization"""
        print_header("Training with QAT")

        # Training hyperparameters
        lr = self.qat_config['learning_rate']
        max_steps = self.qat_config['max_steps']

        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Max steps: {max_steps}")

        # Setup optimizer
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Setup scheduler
        total_steps = min(len(dataloader) * epochs, max_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop
        model.train()
        global_step = 0
        total_loss = 0.0

        print(f"\nStarting training for {epochs} epochs (max {max_steps} steps)...")

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")

            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

            for batch_idx, batch in enumerate(progress_bar):
                if global_step >= max_steps:
                    print(f"\nReached max steps ({max_steps}). Stopping training.")
                    break

                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                try:
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Update metrics
                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    global_step += 1

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{total_loss/global_step:.4f}'
                    })

                    # Log every N steps
                    if global_step % 10 == 0:
                        avg_loss = total_loss / global_step
                        print(f"\nStep {global_step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}")

                except Exception as e:
                    print(f"\nError in training step: {e}")
                    continue

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")

            if global_step >= max_steps:
                break

        print(f"\n✓ Training complete!")
        print(f"  Total steps: {global_step}")
        print(f"  Final average loss: {total_loss/global_step:.4f}")

        return model

    def convert_qat_model(self, model, base_config):
        """Convert fake quantization to real quantization"""
        print_header("Converting QAT Model")

        print("Converting fake quantization to real quantization...")

        # Create convert config
        convert_config = QATConfig(base_config, step="convert")

        # Convert model
        quantize_(model, convert_config)

        print("✓ Model converted to real quantization")

        return model

    def run_qat_pipeline(self, model_config: Dict) -> str:
        """Run complete QAT pipeline"""
        model_name = model_config['name']
        hf_path = model_config['hf_path']

        print_header(f"QAT PIPELINE: {model_name}", char="#")

        output_dir = f"models/{model_name}-qat-int{self.qat_config['bits']}"
        os.makedirs(output_dir, exist_ok=True)

        # Check if already trained
        if os.path.exists(os.path.join(output_dir, "config.json")):
            print(f"QAT model already exists at {output_dir}")
            return output_dir

        try:
            # Step 1: Prepare model for QAT
            model, processor, base_config = self.prepare_qat_model(model_name, hf_path)

            # Step 2: Create dataloader
            dataloader = self.create_dataloader(
                processor,
                batch_size=self.qat_config['batch_size']
            )

            if dataloader is None:
                print("Cannot proceed without training data")
                return None

            # Step 3: Train with fake quantization
            model = self.train_qat(
                model,
                dataloader,
                epochs=self.qat_config['training_epochs']
            )

            # Step 4: Convert to real quantization
            model = self.convert_qat_model(model, base_config)

            # Step 5: Save model
            print(f"\nSaving QAT model to {output_dir}...")
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

            print(f"\n✓ QAT pipeline complete!")
            print(f"  Model saved to: {output_dir}")

            return output_dir

        except Exception as e:
            print(f"\n✗ Error in QAT pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    parser = argparse.ArgumentParser(description="QAT training for VLMs")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to train (default: first model)")
    parser.add_argument("--output", type=str, default="results/qat_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    # Log system info
    log_system_info()

    # Load configuration
    config = load_config(args.config)

    # Filter models if specified
    models_to_train = config['models']
    if args.model:
        models_to_train = [m for m in models_to_train if m['name'] == args.model]
    else:
        # Default: train only the first (highest priority) model
        models_to_train = models_to_train[:1]

    # Initialize trainer
    trainer = QATTrainer(config)

    # Run QAT
    all_results = {}

    for model_config in models_to_train:
        model_name = model_config['name']

        try:
            output_dir = trainer.run_qat_pipeline(model_config)

            if output_dir:
                all_results[model_name] = {
                    "model_path": output_dir,
                    "bits": config['quantization']['qat']['bits'],
                    "method": "qat",
                    "status": "success"
                }
            else:
                all_results[model_name] = {
                    "model_path": None,
                    "method": "qat",
                    "status": "failed"
                }

            # Save results
            save_results(all_results, args.output)

        except Exception as e:
            print(f"\n✗ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save final results
    save_results(all_results, args.output)

    print_header("QAT TRAINING COMPLETE", char="#")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
