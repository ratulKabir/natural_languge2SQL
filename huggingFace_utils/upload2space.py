from huggingface_hub import login
from huggingface_hub import upload_folder
from hf_config import HF_TOKEN

login(HF_TOKEN)

repo_name = "rat45/sql-sft-lora-model"

upload_folder(
    repo_id=repo_name,  # Replace with your Space name
    folder_path="./huggingFace_utils/app",
    repo_type="space",
)