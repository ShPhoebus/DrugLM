# scripts/download_model.py

import os
from huggingface_hub import snapshot_download

def main():
    repo_id = "KurisuTL/DrugLM"

    local_dir = os.path.join(os.path.dirname(__file__), "..", "LM_finetune")

    print(f"Start download models from Hugging Face to {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=local_dir,
        library_name="huggingface_hub"
    )
    print("Download completed!")

if __name__ == "__main__":
    main()
