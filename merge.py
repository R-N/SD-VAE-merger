# sources:
# - https://gist.github.com/ProGamerGov/70061a08e3a2da6e9ed83e145ea24a70
# - https://github.com/CrazyBoyM/merge-models

import copy
import torch
import argparse
from safetensors.torch import load_file, save_file


parser = argparse.ArgumentParser(description="Merge model and vae")
parser.add_argument("model", type=str, help="Path to model")
parser.add_argument("vae", type=str, help="Path to vae")
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, default=None, help="Path to the output file.", required=False)

args = parser.parse_args()

def load_weights(path, device):
  if path.endswith(".safetensors"):
      weights = load_file(path, device)
  else:
      weights = torch.load(path, device)
      weights = weights["state_dict"] if "state_dict" in weights else weights
  
  return weights

def save_weights(weights, path):
  if path.endswith(".safetensors"):
      save_file(weights, path)
  else:
      torch.save({"state_dict": weights}, path) 

# Path to model and VAE files that you want to merge
vae_file_path = args.vae
model_file_path = args.model

device = args.device

# Name to use for new model file
new_model_name = args.output

# Load files
vae_model = load_weights(vae_file_path, device=device)
full_model = load_weights(model_file_path, device=device)

# Check for flattened (merged) models
if 'state_dict' in full_model:
    full_model = full_model["state_dict"]
if 'state_dict' in vae_model:
    vae_model = vae_model["state_dict"]
    
# Replace VAE in model file with new VAE
vae_dict = {k: v for k, v in vae_model.items() if k[0:4] not in ["loss", "mode"]}
for k, _ in vae_dict.items():
    key_name = "first_stage_model." + k
    full_model[key_name] = copy.deepcopy(vae_model[k])

# Save model with new VAE
save_weights(full_model, new_model_name)
