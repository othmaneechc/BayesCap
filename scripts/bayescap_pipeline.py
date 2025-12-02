#!/usr/bin/env python3
"""Standalone pipeline runner for the BayesCap + SRGAN workflow.

This script mirrors the original notebook logic so that organizing datasets,
pretraining on DIV2K, sweeping parameters, and comparing checkpoints can be
performed from the command line. See the --help output for available commands.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
import zipfile
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sure the BayesCap src/ directory is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ds import ImgDset  # type: ignore  # noqa: E402
from losses import TempCombLoss  # type: ignore  # noqa: E402
from networks_SRGAN import BayesCap, Generator  # type: ignore  # noqa: E402
from utils import train_BayesCap  # type: ignore  # noqa: E402


def ensure_three_channels(x: torch.Tensor) -> torch.Tensor:
    added_batch_dim = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        added_batch_dim = True
    if x.dim() != 4:
        raise ValueError(f"Expected tensor with 3 or 4 dims, got shape {tuple(x.shape)}")
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if added_batch_dim:
        x = x.squeeze(0)
    return x


def collate_three_channel(batch):
    lr_items = []
    hr_items = []
    for lr_img, hr_img in batch:
        lr_items.append(ensure_three_channels(lr_img))
        hr_items.append(ensure_three_channels(hr_img))
    return torch.stack(lr_items, dim=0), torch.stack(hr_items, dim=0)


def tensor_psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    sr = sr.clamp(0.0, max_val)
    hr = hr.clamp(0.0, max_val)
    mse = torch.mean((sr - hr) ** 2, dim=(1, 2, 3))
    max_term = 20 * torch.log10(torch.tensor(max_val, device=sr.device))
    return max_term - 10 * torch.log10(mse + eps)


def tensor_ssim(sr: torch.Tensor, hr: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    try:
        from kornia.metrics import ssim as kornia_ssim  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("kornia>=0.6 is required for SSIM computation") from exc

    ssim_map = kornia_ssim(sr, hr, window_size=window_size)
    return ssim_map.mean(dim=(1, 2, 3))


def predictive_uncertainty(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    a_map = 1.0 / (alpha + 1e-5)
    gamma_ratio = torch.exp(torch.lgamma(3.0 / (beta + 1e-2))) / torch.exp(torch.lgamma(1.0 / (beta + 1e-2)))
    u_map = (a_map ** 2) * gamma_ratio
    return u_map.mean(dim=1, keepdim=True)


def reconstruction_error(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    return torch.mean((sr - hr) ** 2, dim=1, keepdim=True)


def compute_uce(unc_map: torch.Tensor, err_map: torch.Tensor, num_bins: int = 20, eps: float = 1e-12) -> float:
    pred_std = torch.sqrt(unc_map.clamp_min(0.0) + eps).flatten()
    true_err = torch.sqrt(err_map.clamp_min(0.0) + eps).flatten()
    if pred_std.numel() == 0:
        return float("nan")
    min_std = pred_std.min()
    max_std = pred_std.max()
    if (max_std - min_std) < eps:
        return torch.abs(pred_std.mean() - true_err.mean()).item()
    bin_edges = torch.linspace(min_std, max_std + eps, steps=num_bins + 1, device=pred_std.device)
    total = pred_std.numel()
    uce = torch.zeros(1, device=pred_std.device)
    for idx in range(num_bins):
        mask = (pred_std >= bin_edges[idx]) & (pred_std < bin_edges[idx + 1])
        count = mask.sum()
        if count == 0:
            continue
        weight = count.float() / total
        mean_unc = pred_std[mask].mean()
        mean_err = true_err[mask].mean()
        uce += weight * torch.abs(mean_unc - mean_err)
    return uce.item()


def corrcoef(unc_map: torch.Tensor, err_map: torch.Tensor, eps: float = 1e-12) -> float:
    pred = unc_map.flatten()
    err = err_map.flatten()
    pred = pred - pred.mean()
    err = err - err.mean()
    pred_std = pred.std(unbiased=False)
    err_std = err.std(unbiased=False)
    if pred_std < eps or err_std < eps:
        return 0.0
    return (pred * err).mean().div(pred_std * err_std).item()


def evaluate_sr_metrics(
    net_g: nn.Module,
    net_c: nn.Module,
    dataset_names: Sequence[str],
    dataset_root: Path,
    image_size: Tuple[int, int] = (256, 256),
    upscale_factor: int = 4,
    batch_size: int = 1,
    num_bins: int = 20,
    device: str = "cuda",
    dtype: torch.dtype | None = torch.cuda.FloatTensor,
) -> pd.DataFrame:
    net_g.eval()
    net_c.eval()
    summary = []
    totals: defaultdict[str, float]
    for name in dataset_names:
        data_dir = dataset_root / name / "original"
        if not data_dir.exists():
            print(f"[skip] {name}: {data_dir} not found")
            continue
        dset = ImgDset(dataroot=str(data_dir), image_size=image_size, upscale_factor=upscale_factor, mode="val")
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_three_channel,
        )
        totals = defaultdict(float)
        samples = 0
        with torch.no_grad():
            for xLR, xHR in loader:
                xLR = ensure_three_channels(xLR)
                xHR = ensure_three_channels(xHR)
                xLR = xLR.to(device)
                xHR = xHR.to(device)
                if dtype is not None:
                    xLR = xLR.type(dtype)
                    xHR = xHR.type(dtype)
                xSR = net_g(xLR)
                xSRC_mu, xSRC_alpha, xSRC_beta = net_c(xSR)
                psnr_vals = tensor_psnr(xSR, xHR)
                ssim_vals = tensor_ssim(xSR, xHR)
                err_map = reconstruction_error(xSR, xHR)
                unc_map = predictive_uncertainty(xSRC_alpha, xSRC_beta)
                for b in range(psnr_vals.shape[0]):
                    totals["psnr"] += psnr_vals[b].item()
                    totals["ssim"] += ssim_vals[b].item()
                    totals["uce"] += compute_uce(unc_map[b : b + 1], err_map[b : b + 1], num_bins=num_bins)
                    totals["corr"] += corrcoef(unc_map[b : b + 1], err_map[b : b + 1])
                    samples += 1
        if samples == 0:
            continue
        summary.append(
            {
                "Dataset": name,
                "PSNR": totals["psnr"] / samples,
                "SSIM": totals["ssim"] / samples,
                "UCE": totals["uce"] / samples,
                "C.Coeff": totals["corr"] / samples,
                "Images": samples,
            }
        )
    if not summary:
        return pd.DataFrame()
    return pd.DataFrame(summary).set_index("Dataset").sort_index()


def organize_sr_benchmarks(data_root: Path, dataset_names: Sequence[str], sr_factor: int = 4) -> None:
    val_root = data_root / "SR" / "val"
    copied = defaultdict(int)
    for name in dataset_names:
        src_dir = data_root / name / f"image_SRF_{sr_factor}"
        if not src_dir.exists():
            print(f"[skip] {src_dir} not found")
            continue
        dest_dir = val_root / name / "original"
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_SRF_{sr_factor}_HR"
        for hr_path in sorted(src_dir.glob("*_HR.*")):
            stem = hr_path.stem
            clean_stem = stem[:-len(suffix)] if stem.endswith(suffix) else stem
            dest_path = dest_dir / f"{clean_stem}{hr_path.suffix}"
            if dest_path.exists():
                continue
            dest_path.write_bytes(hr_path.read_bytes())
            copied[name] += 1
        print(f"[{name}] copied {copied[name]} HR images to {dest_dir}")
        if copied[name] == 0:
            print("  (nothing to do; files already organized)")
    print("Done organizing SR benchmark datasets.")


def download_div2k(data_root: Path, url: str, force: bool = False) -> Path:
    div2k_root = data_root / "DIV2K"
    div2k_root.mkdir(parents=True, exist_ok=True)
    zip_path = div2k_root / "DIV2K_train_HR.zip"
    if zip_path.exists() and not force:
        print(f"[skip] {zip_path} already present")
        return zip_path
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("content-length", 0))
        chunk_size = 1 << 20
        progress = tqdm(total=total, unit="B", unit_scale=True, desc="DIV2K HR")
        with open(zip_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(len(chunk))
        progress.close()
    print(f"[done] Downloaded {zip_path}")
    return zip_path


def extract_div2k(zip_path: Path, force: bool = False) -> Path:
    target_dir = zip_path.parent / "DIV2K_train_HR"
    if target_dir.exists() and not force:
        print(f"[skip] {target_dir} already extracted")
        return target_dir
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(zip_path.parent)
    print(f"[done] Extracted to {target_dir.parent}")
    return target_dir


def build_div2k_loaders(
    hr_dir: Path,
    sr_factor: int = 4,
    train_crop: Tuple[int, int] = (128, 128),
    val_crop: Tuple[int, int] = (256, 256),
    train_batch: int = 8,
    val_batch: int = 4,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_dset = ImgDset(dataroot=str(hr_dir), image_size=train_crop, upscale_factor=sr_factor, mode="tr")
    val_dset = ImgDset(dataroot=str(hr_dir), image_size=val_crop, upscale_factor=sr_factor, mode="val")
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dset,
        batch_size=train_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=val_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def pretrain_srgan_on_div2k(
    generator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    ckpt_path: Path,
    device: torch.device,
) -> Path:
    generator = generator.to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_psnr = -math.inf
    for epoch in range(epochs):
        generator.train()
        epoch_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"[DIV2K][Epoch {epoch + 1}/{epochs}] train",
            leave=False,
            unit="batch",
        )
        for lr_imgs, hr_imgs in train_bar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            optimizer.zero_grad()
            sr_imgs = generator(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_bar.close()
        epoch_loss /= max(len(train_loader), 1)
        scheduler.step()
        generator.eval()
        val_psnr = []
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"[DIV2K][Epoch {epoch + 1}/{epochs}] val",
                leave=False,
                unit="batch",
            )
            for lr_imgs, hr_imgs in val_bar:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                sr_imgs = generator(lr_imgs)
                batch_psnr = tensor_psnr(sr_imgs, hr_imgs).mean().item()
                val_psnr.append(batch_psnr)
                val_bar.set_postfix(psnr=f"{batch_psnr:.2f}")
            val_bar.close()
        avg_psnr = float(np.mean(val_psnr)) if val_psnr else -math.inf
        print(
            f"[DIV2K][Epoch {epoch + 1}/{epochs}] train loss={epoch_loss:.4f}, val PSNR={avg_psnr:.2f} dB"
        )
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(generator.state_dict(), ckpt_path)
            print(f"  -> saved new best checkpoint to {ckpt_path}")
    return ckpt_path


def finetune_bayescap_on_div2k(
    generator_ckpt: Path,
    output_ckpt: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    init_lr: float,
    device: torch.device,
) -> Path:
    net_g = Generator()
    net_g.load_state_dict(torch.load(generator_ckpt, map_location=device))
    net_g.to(device)
    net_g.eval()
    net_c = BayesCap(in_channels=3, out_channels=3).to(device)
    criterion = TempCombLoss(alpha_eps=1e-5, beta_eps=1e-2)
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    train_BayesCap(
        net_c,
        net_g,
        train_loader,
        val_loader,
        Cri=criterion,
        device=str(device),
        dtype=dtype,
        init_lr=init_lr,
        num_epochs=epochs,
        eval_every=1,
        ckpt_path=str(output_ckpt),
    )
    return output_ckpt


def run_parameter_sweep(
    net_g: nn.Module,
    net_c: nn.Module,
    dataset_root: Path,
    dataset_options: Sequence[Sequence[str]],
    image_sizes: Sequence[Tuple[int, int]],
    batch_sizes: Sequence[int],
    num_bins_list: Sequence[int],
    upscale_factor: int,
    device: torch.device,
    dtype: torch.dtype | None,
    manifest_path: Path,
) -> pd.DataFrame:
    sweep_results = []
    config_records = []
    cfg_iter = product(image_sizes, batch_sizes, num_bins_list, dataset_options)
    for cfg_idx, (img_size, batch_size, num_bins, datasets_subset) in enumerate(cfg_iter, start=1):
        cfg_label = f"cfg_{cfg_idx:02d}"
        print(
            f"[{cfg_label}] datasets={datasets_subset}, image_size={img_size}, batch={batch_size}, num_bins={num_bins}"
        )
        df = evaluate_sr_metrics(
            net_g,
            net_c,
            datasets_subset,
            dataset_root=dataset_root,
            image_size=img_size,
            upscale_factor=upscale_factor,
            batch_size=batch_size,
            num_bins=num_bins,
            device=str(device),
            dtype=dtype,
        )
        if df.empty:
            print(f"[{cfg_label}] No samples found; skipping")
            continue
        df = df.reset_index()
        df["Config"] = cfg_label
        df["ImageSize"] = str(img_size)
        df["BatchSize"] = batch_size
        df["NumBins"] = num_bins
        df["Datasets"] = ", ".join(datasets_subset)
        sweep_results.append(df)
        config_records.append(
            {
                "Config": cfg_label,
                "image_size": img_size,
                "batch_size": batch_size,
                "num_bins": num_bins,
                "datasets": list(datasets_subset),
            }
        )
    if sweep_results:
        sweep_df = pd.concat(sweep_results, ignore_index=True)
        sweep_df = sweep_df.set_index(["Config", "Dataset"])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"runs": config_records}, f, indent=2)
        print(f"Saved sweep manifest to {manifest_path}")
        return sweep_df
    print("No sweeps were executed; verify dataset availability.")
    return pd.DataFrame()


def load_model_pair(generator_ckpt: Path, bayescap_ckpt: Path, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    net_g = Generator()
    net_g.load_state_dict(torch.load(generator_ckpt, map_location=device))
    net_g.to(device)
    net_g.eval()
    net_c = BayesCap(in_channels=3, out_channels=3)
    net_c.load_state_dict(torch.load(bayescap_ckpt, map_location=device))
    net_c.to(device)
    net_c.eval()
    return net_g, net_c


def compare_experiments(
    experiments: Sequence[dict],
    dataset_root: Path,
    dataset_names: Sequence[str],
    image_size: Tuple[int, int],
    upscale_factor: int,
    batch_size: int,
    num_bins: int,
    device: torch.device,
    dtype: torch.dtype | None,
) -> pd.DataFrame:
    frames = []
    for exp in experiments:
        label = exp["label"]
        gen_ckpt = Path(exp["generator_ckpt"])
        netc_ckpt = Path(exp["bayescap_ckpt"])
        description = exp.get("description", "")
        if not gen_ckpt.exists() or not netc_ckpt.exists():
            print(f"[skip] {label} missing ckpts ({gen_ckpt}, {netc_ckpt})")
            continue
        net_g, net_c = load_model_pair(gen_ckpt, netc_ckpt, device)
        df = evaluate_sr_metrics(
            net_g,
            net_c,
            dataset_names,
            dataset_root=dataset_root,
            image_size=image_size,
            upscale_factor=upscale_factor,
            batch_size=batch_size,
            num_bins=num_bins,
            device=str(device),
            dtype=dtype,
        )
        if df.empty:
            continue
        df = df.reset_index()
        df["Experiment"] = label
        df["Description"] = description
        frames.append(df)
    if frames:
        combo = pd.concat(frames, ignore_index=True)
        return combo.set_index(["Experiment", "Dataset"]).sort_index()
    print("No experiments were evaluated; ensure checkpoints exist.")
    return pd.DataFrame()


def parse_image_sizes(values: Iterable[str]) -> List[Tuple[int, int]]:
    sizes = []
    for value in values:
        if "x" not in value:
            raise ValueError(f"Invalid image size '{value}'. Expected format HxW, e.g., 256x256")
        h_str, w_str = value.lower().split("x", 1)
        sizes.append((int(h_str), int(w_str)))
    return sizes


def parse_dataset_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_experiment_registry(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("experiments", [])
    if isinstance(data, list):
        return data
    raise ValueError("Experiment registry must be a list or contain an 'experiments' list")


def build_parser() -> argparse.ArgumentParser:
    default_data_root = PROJECT_ROOT / "data"
    parser = argparse.ArgumentParser(description="BayesCap SRGAN experiment runner")
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="Root directory for datasets")
    parser.add_argument("--sr-factor", type=int, default=4, help="Super-resolution upscale factor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    organize = subparsers.add_parser("organize", help="Organize SR benchmark datasets into SR/val structure")
    organize.add_argument(
        "--datasets",
        nargs="+",
        default=["Set5", "Set14", "BSD100", "Urban100"],
        help="Benchmark dataset names to organize",
    )

    dl = subparsers.add_parser("download-div2k", help="Download the DIV2K HR dataset")
    dl.add_argument(
        "--url",
        default="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        help="URL for the DIV2K HR archive",
    )
    dl.add_argument("--force", action="store_true", help="Re-download even if the archive exists")

    extract = subparsers.add_parser("extract-div2k", help="Extract the DIV2K archive")
    extract.add_argument("--force", action="store_true", help="Re-extract even if the folder exists")

    train_cmd = subparsers.add_parser("train-srgan", help="Pretrain/fine-tune SRGAN generator on DIV2K")
    train_cmd.add_argument("--epochs", type=int, default=20)
    train_cmd.add_argument("--lr", type=float, default=5e-5)
    train_cmd.add_argument("--train-batch", type=int, default=8)
    train_cmd.add_argument("--val-batch", type=int, default=4)
    train_cmd.add_argument("--train-crop", default="128x128")
    train_cmd.add_argument("--val-crop", default="256x256")
    train_cmd.add_argument(
        "--ckpt-path",
        type=Path,
        default=PROJECT_ROOT / "ckpt" / "srgan_DIV2K.pth",
        help="Output path for the generator checkpoint",
    )

    finetune = subparsers.add_parser("finetune-bayescap", help="Adapt BayesCap head using DIV2K")
    finetune.add_argument(
        "--generator-ckpt",
        type=Path,
        default=PROJECT_ROOT / "ckpt" / "srgan_DIV2K.pth",
        help="Path to the DIV2K-trained generator checkpoint",
    )
    finetune.add_argument(
        "--output-ckpt",
        type=Path,
        default=PROJECT_ROOT / "ckpt" / "BayesCap_SRGAN_DIV2K.pth",
        help="Output path for the BayesCap checkpoint",
    )
    finetune.add_argument("--epochs", type=int, default=15)
    finetune.add_argument("--lr", type=float, default=5e-5)

    eval_cmd = subparsers.add_parser("evaluate", help="Evaluate one checkpoint pair on benchmark datasets")
    eval_cmd.add_argument("--generator-ckpt", type=Path, required=True)
    eval_cmd.add_argument("--bayescap-ckpt", type=Path, required=True)
    eval_cmd.add_argument(
        "--datasets",
        type=str,
        default="Set5,Set14,BSD100,Urban100",
        help="Comma-separated dataset names",
    )
    eval_cmd.add_argument("--image-size", type=str, default="256x256")
    eval_cmd.add_argument("--batch-size", type=int, default=1)
    eval_cmd.add_argument("--num-bins", type=int, default=30)

    sweep = subparsers.add_parser("sweep", help="Run parameter sweep for a checkpoint pair")
    sweep.add_argument("--generator-ckpt", type=Path, required=True)
    sweep.add_argument("--bayescap-ckpt", type=Path, required=True)
    sweep.add_argument("--image-sizes", nargs="+", default=["128x128", "256x256", "320x320"])
    sweep.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2])
    sweep.add_argument("--num-bins", nargs="+", type=int, default=[15, 20, 30, 40])
    sweep.add_argument(
        "--dataset-options",
        nargs="+",
        default=["Set5,Set14,BSD100", "Set5,Set14,BSD100,Urban100"],
        help="List of dataset groupings (comma-separated per option)",
    )
    sweep.add_argument(
        "--manifest-path",
        type=Path,
        default=default_data_root / "SR" / "val" / "bayescap_sweep_manifest.json",
    )

    compare = subparsers.add_parser("compare", help="Compare multiple experiment checkpoints")
    compare.add_argument("--registry", type=Path, required=True, help="JSON file describing experiments")
    compare.add_argument("--image-size", type=str, default="256x256")
    compare.add_argument("--batch-size", type=int, default=1)
    compare.add_argument("--num-bins", type=int, default=30)
    compare.add_argument(
        "--datasets",
        type=str,
        default="Set5,Set14,BSD100,Urban100",
        help="Comma-separated dataset names",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_root: Path = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if args.command == "organize":
        organize_sr_benchmarks(data_root, args.datasets, sr_factor=args.sr_factor)
        return

    if args.command == "download-div2k":
        download_div2k(data_root, args.url, force=args.force)
        return

    if args.command == "extract-div2k":
        zip_path = data_root / "DIV2K" / "DIV2K_train_HR.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Archive not found at {zip_path}; run download-div2k first")
        extract_div2k(zip_path, force=args.force)
        return

    if args.command in {"train-srgan", "finetune-bayescap"}:
        div2k_hr_dir = data_root / "DIV2K" / "DIV2K_train_HR"
        if not div2k_hr_dir.exists():
            raise FileNotFoundError(f"DIV2K HR directory not found at {div2k_hr_dir}; run extract-div2k first")
        train_crop = tuple(int(x) for x in args.train_crop.lower().split("x")) if "train_crop" in args else (128, 128)
        val_crop = tuple(int(x) for x in args.val_crop.lower().split("x")) if "val_crop" in args else (256, 256)
        train_loader, val_loader = build_div2k_loaders(
            div2k_hr_dir,
            sr_factor=args.sr_factor,
            train_crop=train_crop,
            val_crop=val_crop,
            train_batch=getattr(args, "train_batch", 8),
            val_batch=getattr(args, "val_batch", 4),
        )
        if args.command == "train-srgan":
            generator = Generator()
            pretrain_srgan_on_div2k(
                generator,
                train_loader,
                val_loader,
                epochs=args.epochs,
                lr=args.lr,
                ckpt_path=args.ckpt_path,
                device=device,
            )
            return
        finetune_bayescap_on_div2k(
            args.generator_ckpt,
            args.output_ckpt,
            train_loader,
            val_loader,
            epochs=args.epochs,
            init_lr=args.lr,
            device=device,
        )
        return

    benchmark_root = data_root / "SR" / "val"

    if args.command == "evaluate":
        datasets = parse_dataset_list(args.datasets)
        image_size = parse_image_sizes([args.image_size])[0]
        net_g, net_c = load_model_pair(args.generator_ckpt, args.bayescap_ckpt, device)
        df = evaluate_sr_metrics(
            net_g,
            net_c,
            datasets,
            dataset_root=benchmark_root,
            image_size=image_size,
            upscale_factor=args.sr_factor,
            batch_size=args.batch_size,
            num_bins=args.num_bins,
            device=str(device),
            dtype=dtype,
        )
        print(df)
        return

    if args.command == "sweep":
        datasets_options = [parse_dataset_list(item) for item in args.dataset_options]
        image_sizes = parse_image_sizes(args.image_sizes)
        net_g, net_c = load_model_pair(args.generator_ckpt, args.bayescap_ckpt, device)
        sweep_df = run_parameter_sweep(
            net_g,
            net_c,
            benchmark_root,
            datasets_options,
            image_sizes,
            args.batch_sizes,
            args.num_bins,
            upscale_factor=args.sr_factor,
            device=device,
            dtype=dtype,
            manifest_path=args.manifest_path,
        )
        if not sweep_df.empty:
            print(sweep_df)
        return

    if args.command == "compare":
        experiments = load_experiment_registry(args.registry)
        if not experiments:
            raise ValueError("No experiments found in registry")
        datasets = parse_dataset_list(args.datasets)
        image_size = parse_image_sizes([args.image_size])[0]
        comparison_df = compare_experiments(
            experiments,
            benchmark_root,
            datasets,
            image_size,
            args.sr_factor,
            args.batch_size,
            args.num_bins,
            device,
            dtype,
        )
        if not comparison_df.empty:
            print(comparison_df)
        return


if __name__ == "__main__":
    main()