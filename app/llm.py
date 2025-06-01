import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("llm-server")

model_path = os.getenv("MODEL_PATH", "hf-models/phi-4")
logger.info("Carregando modelo de: %s", model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Dispositivo selecionado: %s", device)

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("pad_token_id não estava definido — configurado como eos_token_id")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

logger.info("Modelo carregado com sucesso.")

def generate_response(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    logger.info("Recebido prompt: %s", prompt)
    logger.debug("Parâmetros de geração: max_tokens=%d, temperature=%.2f", max_tokens, temperature)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        logger.debug("Tokens gerados: %s", output)

        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Texto completo gerado: %s", full_output)

        if full_output.startswith(prompt):
            resposta = full_output[len(prompt):].strip()
        else:
            resposta = full_output.strip()

        logger.info("Resposta final retornada: %s", resposta)
        return resposta

    except Exception as e:
        logger.exception("Erro durante geração de resposta: %s", str(e))
        raise
