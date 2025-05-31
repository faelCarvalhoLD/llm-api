from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "microsoft/phi-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

def generate_response(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature)
    return tokenizer.decode(output[0], skip_special_tokens=True)
