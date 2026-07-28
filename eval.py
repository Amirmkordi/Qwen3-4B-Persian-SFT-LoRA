import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGED_PATH = "./qwen3-4b-merged"
BASE_PATH   = "./qwen3-4b"
DATA_PATH   = "data/dataset.jsonl"
OUTPUT_FILE = "eval_results.txt"
TEST_SIZE   = 0.1
SEED        = 85
MAX_NEW_TOKENS = 512

def load_eval_split():
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    split = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    return split["test"]

def parse_example(text):
    parts = text.split("\nدستیار: ")
    user_msg = parts[0].replace("کاربر: ", "").strip()
    assistant_msg = parts[1].strip() if len(parts) == 2 else ""
    return user_msg, assistant_msg

def build_prompt(user_msg):
    return (
        f"<|im_start|>system\nتو یک دستیار هوشمند دوزبانه هستی.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="mps",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer

def main():
    print("Loading eval split …")
    eval_dataset = load_eval_split()
    examples = [parse_example(ex["text"]) for ex in eval_dataset]
    print(f"  → {len(examples)} eval examples\n")

    models = [
        ("BASE MODEL (qwen3-4b)", BASE_PATH),
        ("FINE-TUNED MODEL (qwen3-4b-merged)", MERGED_PATH),
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("EVALUATION RESULTS\n")
        f.write(f"Total examples: {len(examples)}\n\n")

        for model_label, model_path in models:
            f.write(f"  {model_label}\n")
            model, tokenizer = load_model(model_path)

            for i, (question, expected) in enumerate(examples, 1):
                prompt = build_prompt(question)
                answer = generate_answer(model, tokenizer, prompt)

                f.write(f"--- Example {i} / {len(examples)} ---\n")
                f.write(f"Model     : {model_label}\n")
                f.write(f"Question  : {question}\n")
                f.write(f"Expected  : {expected}\n")
                f.write(f"Generated : {answer}\n")
                f.write("\n")

                if i % 10 == 0:
                    print(f"  {i}/{len(examples)}")
                    f.flush()
            del model
            del tokenizer
            torch.mps.empty_cache()
            print()

if __name__ == "__main__":
    main()