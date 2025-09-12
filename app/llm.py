import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import setup_logger

logger = setup_logger()

model_path = os.getenv("MODEL_PATH", "hf-models/mistral")
logger.info("Carregando modelo de: %s", model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Dispositivo selecionado: %s", device)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("pad_token_id não estava definido — configurado como eos_token_id")

# Escolha do dtype
if device == "cuda" and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
elif device == "cuda":
    dtype = torch.float16
else:
    dtype = torch.float32

# Configuração de memória para evitar OOM
max_memory = {}
if device == "cuda":
    max_memory["cuda:0"] = "12GiB"  # ajuste conforme sua GPU (8–22GiB)
max_memory["cpu"] = "48GiB"         # quanto puder reservar
offload_folder = "./offload"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        offload_folder=offload_folder,
        trust_remote_code=True
    )
except RuntimeError as e:
    logger.warning("Falha ao carregar em GPU, caindo para CPU: %s", str(e))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

model.eval()

# Helper para saber em qual device mandar os inputs
def _first_device_from_map(m):
    d = getattr(m, "hf_device_map", None)
    if not d:
        return device
    for name, dev in d.items():
        if "embed_tokens" in name or "wte" in name:
            return dev
    return next(iter(d.values()))

_first_device = _first_device_from_map(model)
logger.info("Primeiro device para inputs: %s", _first_device)

def generate_response(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    logger.info("Recebido prompt", extra={"prompt": prompt})
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(_first_device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        resposta = generated_text.replace(prompt, "", 1).strip()
        logger.info("Resposta final retornada", extra={"resposta": resposta})
        return resposta
    except Exception as e:
        logger.exception("Erro durante geração de resposta: %s", str(e))
        raise
