import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = os.getenv("MODEL_PATH", "hf-models/phi-4")

print(f"Carregando modelo de: {model_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def generate_response(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature)
    return tokenizer.decode(output[0], skip_special_tokens=True)
