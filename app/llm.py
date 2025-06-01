import os
import logging
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

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
)
model.eval()

logger.info("Modelo carregado com sucesso.")

def generate_response(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    logger.info("Recebido prompt", extra={"prompt": prompt})
    logger.debug("Parâmetros de geração", extra={
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 0.95
    })

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

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

        logger.debug("Tokens gerados", extra={"token_count": len(output[0])})

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Texto completo gerado", extra={"text": generated_text})

        resposta = generated_text.replace(prompt, "").strip()

        logger.info("Resposta final retornada", extra={"resposta": resposta})
        return resposta

    except Exception as e:
        logger.exception("Erro durante geração de resposta: %s", str(e))
        raise
