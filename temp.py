

from huggingface_hub import hf_hub_download
import os



# ----------------------------------------------------------------
# to download specfic generator
# ----------------------------------------------------------------

repo_id = "KKNakka/NAT"
filename = "0_net_G_neuron=109.pth"

# 2. Download to a specific local directory
local_model_path = hf_hub_download(
    repo_id=repo_id, 
    filename=filename,
    local_dir="./checkpoints",  # Forces download to this folder
    local_dir_use_symlinks=False # Ensures the actual file is copied there, not a link
)

print(f"File saved at: {local_model_path}")


# ----------------------------------------------------------------
#  to download all generators
# ----------------------------------------------------------------

repo_id = "KKNakka/NAT"

# 2. Download everything to the ./checkpoints folder
local_dir_path = snapshot_download(
    repo_id=repo_id,
    local_dir="./checkpoints",
    local_dir_use_symlinks=False,
)

print(f"All generators downloaded to: {local_dir_path}")