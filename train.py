import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
import os
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

MODEL_NAME = "./qwen3-4b"
DATA_PATH = "data/dataset.jsonl"
OUTPUT_DIR = "output"
MAX_LENGTH = 128
TRAIN_SPLIT = 0.9
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map=None,
    trust_remote_code=True
)
model = model.to("mps") 
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format(example):
    parts = example["text"].split("\nدستیار: ")
    user_msg = parts[0].replace("کاربر: ", "").strip()
    assistant_msg = parts[1].strip() if len(parts) == 2 else ""
    prompt = (
        f"<|im_start|>system\nتو یک دستیار هوشمند دوزبانه هستی.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    answer = f"{assistant_msg}<|im_end|>"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:MAX_LENGTH]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LENGTH]
    pad_len = MAX_LENGTH - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    labels = labels + [-100] * pad_len
    attention_mask = [1] * (MAX_LENGTH - pad_len) + [0] * pad_len
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

dataset = dataset.map(format, remove_columns=dataset.column_names)
split = dataset.train_test_split(test_size=0.1, seed=85)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch", 
    ),
    train_dataset=split["train"],
    eval_dataset=split["test"],
    data_collator=default_data_collator,
)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)