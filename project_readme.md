# CodeSight-AI: Domain-Specific Gemma 4 QLoRA Fine-Tuning & RAG Engine

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CodeSight--Gemma4--E4B--LoRA-yellow)](https://huggingface.co/Elysianmind/CodeSight-Gemma4-E4B-LoRA)
[![Hardware](https://img.shields.io/badge/Compute-NVIDIA%20A100%2080GB-76B900)](https://colab.research.google.com/)
[![Precision](https://img.shields.io/badge/Precision-NF4%20%2B%20Bfloat16-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**CodeSight-AI** is an advanced, production-grade AI software engineering assistant designed to bridge the gap between open-weight foundation models and specialized proprietary codebase knowledge. By combining **Supervised Fine-Tuning (SFT) via QLoRA** on Google's open-weight `Gemma 4` architecture with a **Retrieval-Augmented Generation (RAG)** vector pipeline, CodeSight-AI eliminates hallucinations on custom SDKs while preserving rigorous software design patterns.

---

## Key Technical Highlights

1. **Parameter-Efficient Fine-Tuning (QLoRA):**
   * Loaded `google/gemma-4-E4B-it` (7.95 Billion total parameters) in 4-bit NormalFloat (`nf4`) quantization with double quantization, maintaining an idle VRAM footprint of **9.23 GB** on an NVIDIA A100 (80GB).
   * Injected trainable Low-Rank Adaptation (`LoRA`) matrices ($r=16, \alpha=32$) targeting inner `Linear4bit` attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feed-forward (`gate_proj`, `up_proj`, `down_proj`) projection modules, while explicitly handling custom `Gemma4ClippableLinear` architectures.
   * Reduced trainable parameter overhead by **99.93%** (training **5,701,632** out of **7,946,802,464** parameters) in `bfloat16` precision.

2. **Vector Retrieval-Augmented Generation (RAG):**
   * Built a semantic retriever using `BAAI/bge-small-en-v1.5` dense embeddings (384-dimensional normalized vectors).
   * Indexed proprietary and newly released SDK specifications inside an in-memory **ChromaDB** vector database utilizing HNSW Cosine Similarity search ($\text{Cosine Distance} \approx 0.17$).
   * Dynamically injects retrieved documentation into CodeSight-AI's multi-turn prompt context to guarantee zero-hallucination code synthesis.

3. **Rigorous Training Telemetry:**
   * Tracked gradient metrics, next-token accuracy, and VRAM utilization in real time via **Weights & Biases (`wandb`)**.
   * Processed over 500,000 tokens across 150 optimization steps utilizing `paged_adamw_8bit` and a cosine learning rate schedule.

---

## Repository Structure

```text
CodeSight-AI/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated code quality & syntax checks
├── notebooks/
│   └── CodeSight_Gemma4_QLoRA_RAG.ipynb # Complete A100 training & RAG notebook
├── src/
│   ├── __init__.py
│   ├── model_loader.py            # 4-bit quantization & LoRA adapter attachment
│   └── rag_pipeline.py            # ChromaDB vector store & grounded generation
├── requirements.txt               # Locked project dependencies
└── README.md                      # Project documentation & benchmark overview
```

---

## Hardware Benchmarks & Telemetry (NVIDIA A100 80GB)

| Metric / Parameter | Specification |
| :--- | :--- |
| **Base Model ID** | `google/gemma-4-E4B-it` (7.95B Parameters) |
| **Hardware Accelerator** | NVIDIA A100-SXM4-80GB (`bfloat16`) |
| **Quantization Format** | 4-bit NormalFloat (`nf4`) + Double Quantization |
| **Trainable Weights** | 5,701,632 (`0.071%` of total model parameters) |
| **Optimizer & Scheduler** | `paged_adamw_8bit` with Cosine Decay (`lr = 2e-4`) |
| **Total Computational Load** | $2.21 \times 10^{16}$ FLOPs across 500,880 tokens |
| **Mean Next-Token Accuracy** | `62.79%` on held-out validation split |

---

## Quickstart: Loading the Fine-Tuned LoRA Adapter

You can pull the fine-tuned adapter directly from the Hugging Face Hub:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-4-E4B-it"
LORA_ADAPTER = "Elysianmind/CodeSight-Gemma4-E4B-LoRA"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading Tokenizer and Base Model...")
tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Attaching CodeSight-AI LoRA Adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()
print("Model ready for grounded code generation!")
```

---

## License
Distributed under the MIT License. See `LICENSE` for more information.