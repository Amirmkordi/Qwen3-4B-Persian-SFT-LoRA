import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model= "./qwen3-4b"
trained= "./output"
final= "./qwen3-4b-merged"
base_model = AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.float16,device_map="cpu")
model= PeftModel.from_pretrained(base_model, trained)
model= model.merge_and_unload()
model.save_pretrained(final)
tokenizer= AutoTokenizer.from_pretrained(trained)
tokenizer.save_pretrained(final)