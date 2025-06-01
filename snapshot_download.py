import argparse
from huggingface_hub import snapshot_download

MODELS = {
    "phi-4": {
        "repo_id": "microsoft/phi-4",
        "cache_dir": "./hf-models/phi-4"
    },
    "phi-2": {
        "repo_id": "microsoft/phi-2",
        "cache_dir": "./hf-models/phi-2"
    },
    "llama3": {
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "cache_dir": "./hf-models/llama3-8b"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Baixa modelos da Hugging Face")
    parser.add_argument(
        "--model",
        choices=MODELS.keys(),
        required=True,
        help="Escolha o modelo: phi-4, phi-2, llama3"
    )
    args = parser.parse_args()

    model_config = MODELS[args.model]
    print(f"Baixando modelo: {args.model} -> {model_config['repo_id']}")
    snapshot_download(
        repo_id=model_config["repo_id"],
        cache_dir=model_config["cache_dir"],
        local_dir_use_symlinks=False
    )
    print("Download concluído.")

if __name__ == "__main__":
    main()
