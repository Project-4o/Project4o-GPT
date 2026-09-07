# Project-4o TODO

## Fine-tuning Pipeline
- [x] Create finetuning branch
- [x] Set up project structure (train.py, merge.py, inference.py, convert_gguf.py)
- [x] Create config.yaml with QLoRA hyperparameters
- [x] Create .gitignore (data/, checkpoints, models)

## Data
- [ ] Collect GPT-4o conversation data (500-5000 conversations per RFC-003)
- [ ] Format data as JSONL in messages format
- [ ] Create data anonymization tooling
- [ ] Set up dataset registry / provenance tracking
- [ ] Create eval dataset for personality/vibe benchmark

## Training
- [ ] Install dependencies (PyTorch ROCm, transformers, trl, peft, bitsandbytes)
- [ ] First training run on MiniCPM5-1B-SFT
- [ ] Evaluate personality quality (vibe check)
- [ ] Iterate on hyperparameters
- [ ] Try different LoRA ranks (16, 32, 64)

## Evaluation
- [ ] Create "vibe benchmark" rubric (per RFC-003)
- [ ] Set up LLM-as-judge evaluation
- [ ] Compare against base MiniCPM5-1B-SFT
- [ ] Compare against GPT-4o reference conversations

## Export & Deployment
- [ ] Merge LoRA adapter into base model
- [ ] Convert to GGUF (q4_k_m, q5_k_m)
- [ ] Test GGUF in llama.cpp / Ollama
- [ ] Create Ollama Modelfile

## Future
- [ ] Scale to Qwen3.5-2B once methodology is proven
- [ ] Explore multimodal (vision) fine-tuning
- [ ] DPO/RLHF for personality alignment
- [ ] Community data collection pipeline
