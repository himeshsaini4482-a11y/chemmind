import os
import random
from datetime import datetime, timedelta
from typing import Optional

from peft import LoraConfig, TaskType
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from supabase import create_client, Client


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADAPTER_OUTPUT_DIR = "./lora_adapters/latest"
PRUNE_AGE_DAYS = 30
PRUNE_NOVELTY_THRESHOLD = 0.3
MIN_TRAINING_SAMPLES = 3


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


def generate_synthetic_data(base_data: list[dict], target_count: int) -> list[dict]:
    prefixes = [
        "Explain the following computational chemistry topic:",
        "Provide detailed guidance on:",
        "What are the best practices for:",
        "How should I approach this simulation problem:",
        "Analyze and provide recommendations for:",
        "Give expert advice regarding:",
    ]
    suffixes = [
        "Include specific parameters and convergence criteria.",
        "Provide step-by-step instructions and common pitfalls.",
        "Include validation methods and expected results.",
        "Discuss theoretical background and practical implementation.",
        "Compare different approaches and recommend the optimal one.",
        "Include typical values for all relevant simulation parameters.",
    ]

    synthetic = list(base_data)
    while len(synthetic) < target_count:
        base = random.choice(base_data)
        content = base["text"].replace("Question: ", "").split("\nAnswer:")[0]
        new_text = f"Question: {random.choice(prefixes)} {content}\nAnswer: {random.choice(suffixes)} Provide expert computational chemistry guidance on this topic."
        synthetic.append({"text": new_text})

    return synthetic


def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )


def train_model(model, tokenizer, training_data: list[dict], lora_config: LoraConfig):
    dataset = Dataset.from_list(training_data)

    sft_config = SFTConfig(
        output_dir=ADAPTER_OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=1,
        report_to="none",
        max_seq_length=512,
    )

    base_model = model.base_model.model if hasattr(model, "base_model") else model

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=lora_config,
    )

    trainer.train()
    model.save_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)


def run_sleep_cycle(model, tokenizer):
    print("Starting sleep cycle...")

    print("Pulling recent memories...")
    memories = pull_recent_memories(limit=50)
    if not memories:
        print("No memories to train on.")
        return

    print(f"Found {len(memories)} memories. Formatting training data...")
    training_data = format_training_data(memories)

    if len(training_data) < MIN_TRAINING_SAMPLES:
        print(f"Only {len(training_data)} samples found. Generating synthetic data to reach {MIN_TRAINING_SAMPLES}...")
        training_data = generate_synthetic_data(training_data, MIN_TRAINING_SAMPLES)

    print("Configuring LoRA adapters...")
    lora_config = get_lora_config()

    print("Starting fine-tuning...")
    train_model(model, tokenizer, training_data, lora_config)

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
