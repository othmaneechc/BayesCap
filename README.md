# BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks

## Project Focus

This repository mirrors the official BayesCap release and documents my workflow for **replicating the DIV2K super-resolution experiments** reported in the paper. The upstream BayesCap team authored the original models, losses, and training utilities inside `src/`, as well as the pretrained checkpoints in `ckpt/`. My contributions are the workflow scaffolding needed for replication:

- `scripts/bayescap_pipeline.py` — command-line wrapper I wrote to orchestrate dataset prep, SRGAN pretraining, BayesCap adaptation, sweeps, and experiment comparison while still invoking the authors' core modules.
- `src/BayesCap_SRGAN_train_and_eval.ipynb` — primary notebook (see below) that walks through the entire reproduction, clearly calling out which cells reuse upstream code and which belong to this replication.
- `results/` and `demo_examples/` assets that I generated while validating the pipeline (training logs, diagnostic plots, visualization grids).

Everything else—`ds.py`, `networks_SRGAN.py`, `utils.py`, BayesCap loss definitions, etc.—remains untouched from the BayesCap authors and is credited to them.

## Environment Setup

```bash
conda create --name bayescap --file requirements.txt
conda activate bayescap
```

The requirements file comes from the original release; no additional packages are needed for my scripts.

## Data Used In This Replication

I only used the datasets below. Each step sticks to the authors' preprocessing so the results remain comparable.

### 1. DIV2K Train HR (for SRGAN and BayesCap training)

```bash
# download + extract using my CLI wrapper
python scripts/bayescap_pipeline.py download-div2k
python scripts/bayescap_pipeline.py extract-div2k
```

Both commands default to `data/DIV2K/`. After extraction you should have `data/DIV2K/DIV2K_train_HR/`. If you prefer manual download, grab `DIV2K_train_HR.zip` from the official [ETH Zürich mirror](http://data.vision.ee.ethz.ch/cvl/DIV2K/) and unzip it to the same path before running the notebook.

### 2. SR Benchmarks (Set5, Set14, BSD100, Urban100) for validation

1. Download the benchmark packs that contain the `image_SRF_4` folders. A convenient mirror that matches the BayesCap preprocessing is the SelfExSR dataset dump: https://github.com/jbhuang0604/SelfExSR/tree/master/data .
2. Place each folder so it matches the following structure:

```
data/
  Set5/image_SRF_4/
  Set14/image_SRF_4/
  BSD100/image_SRF_4/
  Urban100/image_SRF_4/
```

3. Run the organizer so the validation split (`data/SR/val/<dataset>/original`) is populated exactly the way the upstream loaders expect:

```bash
python scripts/bayescap_pipeline.py organize --datasets Set5 Set14 BSD100 Urban100
```

No other datasets (e.g., deblurring, depth) are part of this reproduction, so you can skip downloading them.

## Main Workflow Notebook

The entire reproduction lives in `src/BayesCap_SRGAN_train_and_eval.ipynb`. Every section states whether it executes original BayesCap code or one of my helpers. The notebook covers:

1. Environment bootstrap (imports, device selection, seeding).
2. Benchmark organization and loaders for Set5/Set14/BSD100/Urban100.
3. Baseline evaluation of the ImageNet-pretrained SRGAN + BayesCap checkpoints (authors' release).
4. Experiment comparison + parameter sweeps that I scripted on top of the authors' evaluation utilities.
5. DIV2K download, loader construction, and optional SRGAN/BayesCap training loops (original code wrapped in notebook-friendly helpers).
6. Visualization cell that reuses the BayesCap plotting utilities to compare SR outputs and uncertainty maps.
7. Log parsing/plotting cells that I added to analyze optimization dynamics vs. what's reported in the paper.

Launch Jupyter, open the notebook, and run top-to-bottom. Each markdown block now documents what the subsequent cell does, how long it takes, and whose code it invokes.

## Command-Line Shortcuts

Everything the notebook does can also run headless via:

```bash
# evaluate existing checkpoints on the SR benchmarks
python scripts/bayescap_pipeline.py evaluate \
  --generator-ckpt ckpt/srgan-ImageNet-bc347d67.pth \
  --bayescap-ckpt ckpt/BayesCap_SRGAN_best.pth

# fine-tune SRGAN on DIV2K and adapt BayesCap
python scripts/bayescap_pipeline.py train-srgan --epochs 100 --lr 5e-5
python scripts/bayescap_pipeline.py finetune-bayescap --epochs 50 --lr 5e-5

# sweep scoring hyperparameters for a checkpoint pair
python scripts/bayescap_pipeline.py sweep --generator-ckpt <path> --bayescap-ckpt <path>
```

These commands still defer to the original training/eval code; my wrapper simply wires together the right arguments for reproducibility.

## References

- Paper: [ArXiv](https://arxiv.org/pdf/2207.06873.pdf) · [Blog](https://www.eml-unitue.de/publication/BayesCap) · [HuggingFace Demo](https://huggingface.co/spaces/udion/BayesCap)
- Base model repos referenced by the authors:
  - SRGAN: https://github.com/Lornatang/SRGAN-PyTorch
  - DeepFillv2: https://github.com/csqiangwen/DeepFillv2_Pytorch
  - DeblurGANv2: https://github.com/VITA-Group/DeblurGANv2
  - Medical Image Translation: https://github.com/ExplainableML/UncerGuidedI2I
  - MonoDepth2: https://github.com/nianticlabs/monodepth2

## Citation

Please cite the authors' work when using BayesCap:

```
@inproceedings{Upa_bayescap,
  title = {BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks},
  author = {Upadhyay, U. and Karthik, S. and Chen, Y. and Mancini, M. and Akata, Z.},
  booktitle = {European Conference on Computer Vision (ECCV 2022)},
  year = {2022}
}
```

```
@inproceedings{upadhyay2021uncerguidedi2i,
  title={Uncertainty Guided Progressive GANs for Medical Image Translation},
  author={Upadhyay, Uddeshya and Chen, Yanbei and Hebb, Tobias and Gatidis, Sergios and Akata, Zeynep},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2021},
  organization={Springer}
}
```

```
@inproceedings{UpaCheAka21,
  title = {Robustness via Uncertainty-aware Cycle Consistency},
  author = {Upadhyay, U. and Chen, Y. and Akata, Z.},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS 2021)},
  year = {2021}
}
```


