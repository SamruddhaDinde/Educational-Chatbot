"""
QLoRA fine-tuning of microsoft/Phi-4-mini-instruct on OOP Q&A data.
Run this on the HPC (requires CUDA + ~8 GB VRAM).


"""

import os
import json
import torch
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "app", ".env"))

# ── Paths & hyperparameters ────────────────────────────────────────────────────
MODEL_ID   = "microsoft/Phi-4-mini-instruct"
HF_TOKEN   = os.environ.get("HF_TOKEN")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "oop_qa.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "oop-phi4-lora")

LORA_R       = 16       # adapter rank — higher = more capacity, more memory
LORA_ALPHA   = 32       # scaling factor (rule of thumb: 2 × r)
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 512
NUM_EPOCHS   = 20
BATCH_SIZE   = 2        # per GPU; effective batch = BATCH_SIZE × GRAD_ACCUM
GRAD_ACCUM   = 4        # simulates batch size of 8 without extra VRAM


def load_and_format_dataset(tokenizer) -> Dataset:
    with open(DATA_PATH) as f:
        records = json.load(f)

    # Apply the model's own chat template to each conversation so the training
    # text matches exactly what the model sees at inference time.
    formatted = [
        {"text": tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False
        )}
        for r in records
    ]
    return Dataset.from_list(formatted)


def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # second quantization saves ~0.4 bits/param
        bnb_4bit_quant_type="nf4",         # NormalFloat4 is better than int4 for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"      # required for causal LM training
    tokenizer.model_max_length = MAX_SEQ_LEN

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False        # required when using gradient checkpointing

    return model, tokenizer


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        # These four projection matrices are the standard LoRA injection points
        # for transformer attention layers.
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def train():
    if not torch.cuda.is_available():
        raise EnvironmentError("This script requires a CUDA GPU. Run it on the HPC.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model, tokenizer = load_base_model()
    dataset = load_and_format_dataset(tokenizer)

    print(f"Training samples: {len(dataset)}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,      # trades compute for VRAM savings
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=1,                
        save_strategy="epoch",            # saves a checkpoint after each epoch
        report_to="none",                 # change to "wandb" for live dashboard logging
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_lora_config(),
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    
    log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    loss_history = [
        {"step": entry["step"], "loss": entry["loss"]}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    with open(log_path, "w") as f:
        json.dump(loss_history, f, indent=2)

    print(f"Adapter saved  → {OUTPUT_DIR}")
    print(f"Training log   → {log_path}")


if __name__ == "__main__":
    train()
