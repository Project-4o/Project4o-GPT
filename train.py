"""
Project-4o Fine-tuning Script
QLoRA training on a 16GB AMD ROCm GPU. Set the base model in config.yaml.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --data data/train.jsonl
"""

import argparse
import json
import os
import sys

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(cfg: dict):
    """Load training data from JSONL. Expected format per line:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Or ShareGPT format:
        {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """
    data_cfg = cfg["data"]
    train_path = data_cfg["train_file"]

    if not os.path.exists(train_path):
        print(f"ERROR: Training data not found at {train_path}")
        print("Expected JSONL format with 'messages' or 'conversations' field.")
        print("See docs/PROJECT.md for data format details.")
        sys.exit(1)

    raw_data = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))

    if len(raw_data) < data_cfg.get("min_samples", 100):
        print(f"WARNING: Only {len(raw_data)} samples found (minimum: {data_cfg.get('min_samples', 100)})")

    dataset = Dataset.from_list(raw_data)

    eval_path = data_cfg.get("eval_file", "")
    if os.path.exists(eval_path):
        eval_data = []
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_data.append(json.loads(line))
        eval_dataset = Dataset.from_list(eval_data)
    else:
        split = dataset.train_test_split(test_size=data_cfg["eval_split"], seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]

    print(f"Train samples: {len(dataset)}")
    print(f"Eval samples:  {len(eval_dataset)}")
    return dataset, eval_dataset


def format_conversation(example, tokenizer):
    """Format a conversation into text using the model's chat template."""
    messages = example.get("messages")
    if messages is None:
        conversations = example.get("conversations", [])
        messages = []
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Project-4o QLoRA fine-tuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", type=str, default=None, help="Override training data path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data:
        cfg["data"]["train_file"] = args.data

    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    if not model_cfg.get("base_model"):
        print("ERROR: model.base_model is not set in config.yaml")
        print("Set it once the community chooses a model (see RFC-003 / r/project4o megathread)")
        return

    print("=" * 60)
    print("Project-4o Fine-tuning")
    print(f"Base model: {model_cfg['base_model']}")
    print(f"Method: QLoRA (4-bit) + LoRA (r={lora_cfg['r']})")
    print("=" * 60)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model with 4-bit quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=getattr(torch, model_cfg["model_dtype"]),
    )

    model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load data ---
    train_dataset, eval_dataset = load_dataset(cfg)

    train_dataset = train_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=eval_dataset.column_names,
    )

    # --- Training arguments ---
    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        max_seq_length=train_cfg["max_seq_length"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        report_to=train_cfg["report_to"],
        eval_strategy="steps",
        eval_steps=train_cfg["save_steps"],
        save_strategy="steps",
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # --- Train ---
    print("\nStarting training...")
    trainer.train()

    # --- Save adapter ---
    adapter_path = os.path.join(train_cfg["output_dir"], "final_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to {adapter_path}")
    print("Next step: python merge.py --config config.yaml")


if __name__ == "__main__":
    main()
