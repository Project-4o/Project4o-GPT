# Project-4o TODO

## Fine-tuning Pipeline
- [x] Create finetuning branch
- [x] Set up project structure (train.py, merge.py, inference.py, convert_gguf.py)
- [x] Create config.yaml with QLoRA hyperparameters
- [x] Create .gitignore (data/, checkpoints, models)

## Model Selection
- [ ] Community discussion on base model — [r/project4o megathread](https://www.reddit.com/r/project4o/comments/1wcteap/megathread_which_model/)
- [ ] Set `base_model` in config.yaml once chosen

## Data
- [ ] Collect GPT-4o conversation data (500-5000 conversations per RFC-003)
- [ ] Format data as JSONL in messages format
- [ ] Create data anonymization tooling
- [ ] Set up dataset registry / provenance tracking
- [ ] Create eval dataset for personality/vibe benchmark

## Training
- [ ] Install dependencies (PyTorch ROCm, transformers, trl, peft, bitsandbytes)
- [ ] First training run on the chosen base model
- [ ] Evaluate personality quality (vibe check)
- [ ] Iterate on hyperparameters
- [ ] Try different LoRA ranks (16, 32, 64)

## Evaluation
- [ ] Create "vibe benchmark" rubric (per RFC-003)
- [ ] Set up LLM-as-judge evaluation
- [ ] Compare against the untrained base model
- [ ] Compare against GPT-4o reference conversations

## Export & Deployment
- [ ] Merge LoRA adapter into base model
- [ ] Convert to GGUF (q4_k_m, q5_k_m)
- [ ] Test GGUF in llama.cpp / Ollama
- [ ] Create Ollama Modelfile

## Future
- [ ] Scale to a larger model once methodology is proven
- [ ] Explore multimodal (vision) fine-tuning
- [ ] DPO/RLHF for personality alignment
- [ ] Community data collection pipeline
