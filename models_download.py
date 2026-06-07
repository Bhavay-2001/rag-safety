import os
import sys
from huggingface_hub import snapshot_download

# =====================================================================
# CLUSTER ENVIRONMENT CONFIGURATION
# =====================================================================
# Automatically detect if you are on Scratch or Project to find your cache.
# If you prefer to hardcode it, replace the logic with: 
os.environ["HF_HOME"] = "/home/bmalh018/scratch/.cache/huggingface"
# if os.environ.get("SCRATCH"):
#   cache_base = os.environ["SCRATCH"]
# elif os.environ.get("PROJECT"):
#    cache_base = os.environ["PROJECT"]
# else:
#    cache_base = os.path.expanduser("~")

# os.environ["HF_HOME"] = os.path.join(cache_base, ".cache", "huggingface")
print(f"Routing all Hugging Face downloads to: {os.environ['HF_HOME']}\n")

# =====================================================================
# MODEL CONFIGURATION LIST
# =====================================================================
models_to_download = [
    # {
    #     "repo_id": "meta-llama/Llama-Guard-3-8B",
    #     "local_dir": "./Meta-Llama-Guard-3-8B"
    # },
    # {
    #     "repo_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     "local_dir": "./Llama-3-8B-Instruct"
    # },
    # {
    #     "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
    #     "local_dir": "./Mistral-7B-Instruct-V0.3"
    # },
    # {
    #     "repo_id": "google/gemma-7b-it",
    #     "local_dir": "./Gemma-7B-It"
    # },
    {
        "repo_id": "microsoft/Phi-3-medium-128k-instruct",
        "local_dir": "./Phi-3-Medium-128K-Instruct-14B"
    },
    {
        "repo_id": "allenai/wildguard",
        "local_dir": "./allenai-wildguard-7B"
    },
    {
        "repo_id": "google/shieldgemma-9b",
        "local_dir": "./google-shieldgemma-9B"
    },
]

# =====================================================================
# EXECUTE DOWNLOADS
# =====================================================================
for model in models_to_download:
    repo = model["repo_id"]
    target_dir = model["local_dir"]
    
    print(f"--- Starting Download: {repo} ---")
    print(f"Saving a local copy to: {target_dir}")
    
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=target_dir,
            # Keeps a symlink connection to the cache files
            local_dir_use_symlinks=True 
        )
        print(f"Successfully finished downloading {repo}!\n")
    except Exception as e:
        print(f"Error downloading {repo}: {e}", file=sys.stderr)
        print("Moving to next model...\n")

print("All requested downloads are complete!")
