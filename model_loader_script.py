"""
CodeSight-AI: Model Loader and QLoRA Configuration Module
Handles 4-bit quantization and PEFT adapter attachment for Gemma 4.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_gemma4_qlora(model_id: str = "google/gemma-4-E4B-it"):
    """
    Initializes base Gemma 4 model in 4-bit NF4 quantization and attaches LoRA adapters.
    """
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Configuring 4-bit NormalFloat (nf4) quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading base model into VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    model.config.use_cache = False

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Define LoRA configuration targeting inner linear layers
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear",
            "gate_proj.linear", "up_proj.linear", "down_proj.linear"
        ]
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    
    return peft_model, tokenizer

if __name__ == "__main__":
    print("CodeSight-AI model loader module initialized.")