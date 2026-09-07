"""
Project-4o Inference Script
Quick interactive chat to test the fine-tuned model.

Usage:
    python inference.py --config config.yaml                    # Use merged model
    python inference.py --config config.yaml --adapter          # Use LoRA adapter on quantized base
    python inference.py --config config.yaml --model ./merged_model  # Custom model path
"""

import argparse
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Project-4o inference test")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Model path or HF ID")
    parser.add_argument("--adapter", action="store_true", help="Load LoRA adapter on quantized base")
    parser.add_argument("--quantize", action="store_true", help="Load in 4-bit for inference")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.adapter:
        # Load base in 4-bit + adapter
        from peft import PeftModel

        base_model = cfg["model"]["base_model"]
        adapter_path = os.path.join(cfg["training"]["output_dir"], "final_adapter")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading base model {base_model} in 4-bit...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # Load merged model
        model_path = args.model or cfg["merge"]["output_dir"]
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            print("Run merge.py first, or specify --model <path>")
            return

        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if args.quantize:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # --- Interactive chat ---
    print("\n" + "=" * 60)
    print("Project-4o Chat (type 'quit' to exit, 'clear' to reset)")
    print("=" * 60 + "\n")

    system_prompt = "You are Project-4o, a friendly and natural AI assistant with a warm, conversational personality."
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("(conversation cleared)\n")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.9,
                top_k=20,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"4o: {response}\n")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
