import os
from typing import Optional

import torch
from unsloth import FastModel
from transformers import TextStreamer


MODEL_ID = "unsloth/Qwen3.5-9B-bnb-4bit"
MAX_SEQ_LENGTH = 4096
DTYPE = torch.float16
LOAD_4BIT = True


_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    _model, _tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_4BIT,
    )

    _model = FastModel.get_peft_model(
        _model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    for param in _model.base_model.parameters():
        param.requires_grad = False

    _model.eval()
    return _model, _tokenizer


def generate(
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stream: bool = False,
) -> str:
    model, tokenizer = load_model()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker in response:
        response = response.split(assistant_marker)[-1].strip()
    return response
