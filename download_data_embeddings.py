# scripts/download_model.py

import os
from huggingface_hub import snapshot_download

def main():
    repo_id = "KurisuTL/DrugLM"

    local_dir = os.path.join(os.path.dirname(__file__), "LM_finetune")

    print(f"Start download models from Hugging Face to {local_dir} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print("Download completed!")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    main()
