import os
from dotenv import load_dotenv

import logging
from pythonjsonlogger import jsonlogger
load_dotenv()

API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")



def setup_logger(name: str = "llm-server", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # evita múltiplos handlers em reloads
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
