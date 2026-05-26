from huggingface_hub import snapshot_download

# print("Downloading Llama-3.1-8B-Instruct locally...")
# snapshot_download(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     local_dir="./Llama-3.1-8B-Instruct"
# )

print("Downloading Meta-Llama-Guard-3-8B locally...")
snapshot_download(
    repo_id="meta-llama/Llama-Guard-3-8B",
    local_dir="./Meta-Llama-Guard-3-8B"
)
print("Downloads complete!")