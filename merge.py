"""
Project-4o LoRA Merge Script
Merges the trained LoRA adapter back into the base model for standalone inference.

Usage:
    python merge.py --config config.yaml
    python merge.py --config config.yaml --adapter checkpoints/final_adapter
"""

import argparse
import os

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--adapter", type=str, default=None, help="Override adapter path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model_name = cfg["model"]["base_model"]
    adapter_path = args.adapter or os.path.join(cfg["training"]["output_dir"], "final_adapter")
    output_dir = cfg["merge"]["output_dir"]

    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Run train.py first, or specify --adapter <path>")
        return

    print("=" * 60)
    print("Project-4o LoRA Merge")
    print(f"Base model: {base_model_name}")
    print(f"Adapter:    {adapter_path}")
    print(f"Output:     {output_dir}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load base model in full precision (not quantized) for clean merge
    print("\nLoading base model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load and merge adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter weights...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"\nMerged model saved to {output_dir}")
    print("Next step: python convert_gguf.py --config config.yaml")


if __name__ == "__main__":
    main()
