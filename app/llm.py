import os
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import setup_logger

logger = setup_logger()
MODEL_PATH = os.getenv("MODEL_PATH", "hf-models/mistral")
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "./offload")
DEFAULT_CPU_MEM = os.getenv("MAX_CPU_MEM", "48GiB")
DEFAULT_GPU0_MEM = os.getenv("MAX_GPU_0_MEM", "20GiB")

logger.info("Carregando modelo de: %s", MODEL_PATH)
cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
logger.info("Dispositivo selecionado: %s", device)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("pad_token_id não estava definido — configurado como eos_token_id")

if cuda_available and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
elif cuda_available:
    dtype = torch.float16
else:
    dtype = torch.float32
logger.info("dtype selecionado: %s", str(dtype).replace("torch.", ""))

max_memory = {"cpu": DEFAULT_CPU_MEM}
if cuda_available:
    try:
        gpu_count = torch.cuda.device_count()
    except Exception:
        gpu_count = 0
    if gpu_count > 0:
        max_memory[0] = DEFAULT_GPU0_MEM
        for i in range(1, gpu_count):
            env_key = f"MAX_GPU_{i}_MEM"
            if env_key in os.environ:
                max_memory[i] = os.environ[env_key]
            else:
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024**3)
                usable = max(4, math.floor(total_gb * 0.88))
                max_memory[i] = f"{usable}GiB"
logger.info("max_memory: %s", max_memory)

os.makedirs(OFFLOAD_DIR, exist_ok=True)

def _load_model():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            dtype=dtype,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            offload_folder=OFFLOAD_DIR,
            trust_remote_code=True,
        )
        logger.info("Modelo carregado com device_map=auto e dtype=%s", str(dtype).replace("torch.", ""))
        return model
    except Exception as e:
        logger.warning("Falha ao carregar em GPU, caindo para CPU: %s", str(e))
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map={"": "cpu"},
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        logger.info("Modelo carregado na CPU com float32")
        return model

model = _load_model()
model.eval()

# ============================================================
# PATCH: normalização robusta de devices
# ============================================================

def _normalize_device(dev):
    if isinstance(dev, torch.device):
        return dev

    if isinstance(dev, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{dev}")
        return torch.device("cpu")

    if isinstance(dev, str):
        s = dev.strip().lower()
        if s in {"disk", "meta", "offload"}:
            return torch.device("cpu")
        if s.isdigit():
            if torch.cuda.is_available():
                return torch.device(f"cuda:{s}")
            return torch.device("cpu")
        if s == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        if s.startswith("cuda"):
            return torch.device(s if torch.cuda.is_available() else "cpu")
        if s == "cpu":
            return torch.device("cpu")
        return torch.device("cpu")

    return torch.device("cpu")


def _first_device_from_map(m):
    dmap = getattr(m, "hf_device_map", None)
    if not dmap:
        return _normalize_device("cuda:0" if torch.cuda.is_available() else "cpu")

    for name, dev in dmap.items():
        if "embed_tokens" in name or "wte" in name:
            return _normalize_device(dev)

    for dev in dmap.values():
        if str(dev).lower() not in {"disk", "meta", "offload"}:
            return _normalize_device(dev)

    return torch.device("cpu")


_first_device = _first_device_from_map(model)
logger.info("Primeiro device para inputs: %s", str(_first_device))


def _to_device(batch, dev):
    tdev = _normalize_device(dev)
    return {k: (v.to(tdev) if hasattr(v, "to") else v) for k, v in batch.items()}

# ============================================================
# Função principal de geração
# ============================================================

def generate_response(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    logger.info("Recebido prompt", extra={"prompt": prompt[:500] + ("..." if len(prompt) > 500 else "")})
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = _to_device(inputs, _first_device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=float(temperature),
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        resposta = generated_text
        if generated_text.startswith(prompt):
            resposta = generated_text[len(prompt):].lstrip()
        logger.info("Resposta final retornada", extra={"resposta": resposta[:500] + ("..." if len(resposta) > 500 else "")})
        return resposta
    except Exception as e:
        logger.exception("Erro durante geração de resposta: %s", str(e))
        raise
