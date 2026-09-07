# Project-4o

## Overview

Project-4o is an open-source project to build a 4o-like AI, focused on personality, not parameters. The goal is natural, fluid conversation with real personality and working roleplay mode.

## Current Architecture (finetuning branch)

### Base Model
- **openbmb/MiniCPM5-1B-SFT** — 1B parameter dense model, SFT checkpoint
- Standard `LlamaForCausalLM` architecture, 131K context
- Chosen for: small size (fits 16GB VRAM), SFT stage (clean slate for personality), MIT license

### Fine-tuning Method
- **QLoRA** (4-bit NF4 quantization + LoRA adapters)
- LoRA rank 32, alpha 64, dropout 0.05
- All attention + MLP modules targeted
- Training: TRL SFTTrainer, bf16, gradient checkpointing
- Hardware target: AMD ROCm 16GB VRAM

### Pipeline
```
data/train.jsonl → train.py → checkpoints/final_adapter (LoRA)
                                          ↓
                                     merge.py → merged_model/ (full weights)
                                          ↓
                                 convert_gguf.py → gguf_output/ (GGUF file)
```

### Scripts
| Script | Purpose |
|--------|---------|
| `train.py` | QLoRA fine-tuning with SFTTrainer |
| `merge.py` | Merge LoRA adapter into base model |
| `inference.py` | Interactive chat test |
| `convert_gguf.py` | Export to GGUF for llama.cpp/Ollama |

### Data Format
Training data goes in `data/train.jsonl` (gitignored). Two supported formats:

**Messages format (preferred):**
```json
{"messages": [{"role": "system", "content": "You are Project-4o..."}, {"role": "user", "content": "Hey!"}, {"role": "assistant", "content": "Hey! What's up?"}]}
```

**ShareGPT format:**
```json
{"conversations": [{"from": "human", "value": "Hey!"}, {"from": "gpt", "value": "Hey! What's up?"}]}
```

### Configuration
All hyperparameters in `config.yaml`. Override data path with `--data` flag.

## Design Direction
- Start small (1B), prove personality fine-tuning works
- Iterate on data quality over model size
- Target: GPT-4o-like conversational vibe — warm, natural, not corporate
- Eventually scale to 2B-3B models once methodology is proven
- Export to GGUF for broad deployment (llama.cpp, Ollama, LM Studio)
