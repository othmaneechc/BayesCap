# BayesCap SRGAN Pipeline Script

`bayescap_pipeline.py` mirrors the original notebook logic while exposing a
command-line interface that covers dataset organisation, DIV2K pretraining, and
benchmark evaluation. Example usage patterns:

```bash
# Organise benchmark HR images (Set5/Set14/BSD100/Urban100)
python scripts/bayescap_pipeline.py organize --data-root /data/oe23/BayesCap/data

# Download + extract DIV2K
python scripts/bayescap_pipeline.py download-div2k
python scripts/bayescap_pipeline.py extract-div2k

# Train generator on DIV2K and fine-tune BayesCap
python scripts/bayescap_pipeline.py train-srgan --epochs 20 --lr 5e-5
python scripts/bayescap_pipeline.py finetune-bayescap --epochs 15 --lr 5e-5

# Evaluate ImageNet vs DIV2K checkpoints (registry JSON)
python scripts/bayescap_pipeline.py compare --registry scripts/experiment_registry.json

# Run parameter sweep for a checkpoint pair
python scripts/bayescap_pipeline.py sweep \
  --generator-ckpt ckpt/srgan-ImageNet-bc347d67.pth \
  --bayescap-ckpt ckpt/BayesCap_SRGAN_best.pth
```

Use `python scripts/bayescap_pipeline.py --help` to inspect the full set of
commands and options.
