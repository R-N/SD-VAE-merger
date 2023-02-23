# SD-VAE-merger
Script to merge VAE into checkpoint (untested)

Usage:
```
python merge.py mymodel.ckpt mymodel.vae.pt --device cpu --output merged.safetensors
```

Sources:
- https://gist.github.com/ProGamerGov/70061a08e3a2da6e9ed83e145ea24a70
- https://github.com/CrazyBoyM/merge-models/blob/main/merge.py
