import os
import json
from datetime import datetime, timedelta
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from supabase import create_client, Client


MODEL_ID = "Qwen/Qwen3-4B"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADAPTER_OUTPUT_DIR = "./lora_adapters/latest"
PRUNE_AGE_DAYS = 30
PRUNE_NOVELTY_THRESHOLD = 0.3


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def pull_recent_memories(limit: int = 50) -> list[dict]:
    cutoff = (datetime.utcnow() - timedelta(days=PRUNE_AGE_DAYS)).isoformat()

    result = (
        supabase.table("memories")
        .select("content")
        .neq("memory_type", "qa_pair")
        .gte("created_at", cutoff)
        .limit(limit)
        .execute()
    )

    return result.data


def format_training_data(memories: list[dict]) -> list[dict]:
    training_data = []
    for mem in memories:
        content = mem.get("content", "")
        if not content:
            continue
        entry = {
            "text": f"Question: {content}\nAnswer: Provide expert computational chemistry guidance on this topic."
        }
        training_data.append(entry)
    return training_data


def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    return get_peft_model(model, lora_config)


def tokenize_dataset(dataset: Dataset, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def train_model(model, tokenizer, training_data: list[dict]):
    dataset = Dataset.from_list(training_data)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=ADAPTER_OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)


def run_sleep_cycle():
    print("Starting sleep cycle...")

    print("Pulling recent memories...")
    memories = pull_recent_memories(limit=50)
    if not memories:
        print("No memories to train on.")
        return

    print(f"Found {len(memories)} memories. Formatting training data...")
    training_data = format_training_data(memories)
    if not training_data:
        print("No valid training data.")
        return

    print("Loading base model with 4bit quantization...")
    model, tokenizer = load_base_model()

    print("Applying LoRA adapters...")
    model = apply_lora(model)
    model.print_trainable_parameters()

    print("Starting fine-tuning...")
    train_model(model, tokenizer, training_data)

    print(f"Sleep cycle complete. Adapter saved to {ADAPTER_OUTPUT_DIR}")


def prune_memories():
    cutoff = (datetime.utcnow() - timedelta(days=PRUNE_AGE_DAYS)).isoformat()

    result = (
        supabase.table("memories")
        .delete()
        .lt("novelty_score", PRUNE_NOVELTY_THRESHOLD)
        .lt("created_at", cutoff)
        .execute()
    )

    deleted_count = len(result.data) if result.data else 0
    print(f"Pruned {deleted_count} old low-novelty memories.")
    return deleted_count
