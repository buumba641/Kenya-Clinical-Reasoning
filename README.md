# Kenya Clinical Reasoning Challenge – Low‑Resource NLP for Healthcare

[![Zindi](https://img.shields.io/badge/Zindi-Kenya%20Clinical%20Reasoning-blue)](https://zindi.africa/competitions/kenya-clinical-reasoning-challenge)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.46+-orange)](https://huggingface.co/docs/transformers)

This repository contains my solution for the **Kenya Clinical Reasoning Challenge** on Zindi.  
The goal is to predict a clinician’s free‑text response given a clinical vignette (nurse background + patient presentation). The evaluation metric is **ROUGE score**, and solutions must respect strict deployment constraints (≤1B parameters, <100ms inference, <2GB RAM).

**Final scores:**  
- Public leaderboard: **0.35075**  
- Private leaderboard: **0.35945**  
- Benchmark (random/naive): 0.00271 / 0.00362

---

## 📖 Problem Overview

Frontline healthcare workers in rural Kenya face complex decisions with limited specialist support.  
The dataset consists of ~400 training and 100 test examples – authentic clinical prompts paired with expert clinician responses.  
The challenge simulates real‑world constraints: fast, accurate, and deployable on edge devices (e.g., NVIDIA Jetson Nano).

**Key constraints:**  
- Model parameters ≤ 1 billion  
- Inference time < 100 ms per vignette  
- Inference RAM < 2 GB  
- Quantization required  
- Training ≤ 24 hours on T4 GPU

---

## 🧠 Approach

### 1. Model Architecture
- **Base model:** `t5-base` (222M parameters) – well under the 1B limit.  
- Fine‑tuned as a sequence‑to‑sequence task:  
  Input: `"summarize: " + enhanced_prompt`  
  Output: clinician’s summary/assessment.

### 2. Preprocessing & Prompt Engineering
- Cleaned text (standardised county names, removed problematic IDs).  
- Created an **enhanced prompt** by injecting metadata (nursing competency, clinical panel, years of experience) and patient age/gender when available.  
- All inputs prefixed with `"summarize: "` to align with T5’s pre‑training.

### 3. Training Strategy
- **Stratified split** (80/20) based on clinician response length to preserve diversity.  
- Gradient accumulation (batch size 4, accumulation steps 8) to fit in T4 memory.  
- FP16 mixed precision.  
- Optimizer: AdamW (lr=3e-4) with cosine warmup.  
- Early stopping based on validation ROUGE‑L.

### 4. Inference Optimisation (for <100ms)
- **Quantisation:** Model loaded in FP16, then exported to `torch.float16`.  
- **Beam search reduced** from 4 → 1 (greedy) for speed.  
- **Batch inference** with dynamic batch sizing (up to 64 on T4).  
- **torch.compile** (mode="reduce‑overhead") and CUDA TF32 enabled.  
- Final average inference time: **~81 ms per vignette** (well under 100ms).

### 5. Evaluation Metrics
- ROUGE‑1, ROUGE‑2, ROUGE‑L, and BLEU (for reference).  
- Best validation ROUGE‑L: **0.3313**.  
- The gap to public/private scores (~0.35‑0.36) indicates good generalisation.

---

## 📁 Repository Structure
├── submission.csv # Final test predictions
├── Kenya_Clinical_Reasoning.ipynb # Full training & inference notebook
├── README.md # This file
└── LICENSE
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
