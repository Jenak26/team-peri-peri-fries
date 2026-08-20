"""Stage B — tamper mask + reliability map on RGB concatenated with the Videoprint.

Input is six channels: the RGB frame and the Stage A fingerprint field stacked
together. If no Stage A checkpoint exists, the fingerprint channels are filled by
the fixed SRM acquisition residual instead. That fallback is deliberate and is the
hour-14 kill switch: Stage B is trainable and shippable before Stage A finishes,
and Stage A is a hot-swap upgrade rather than a dependency.

Two outputs:
  channel 0  tamper logit         -> the pixel mask
  channel 1  confidence logit     -> the reliability map

The confidence head is trained TCP-style: its target is the model's own predicted
probability for the true class, detached. It learns where the mask head deserves to
be believed. The frontend greys out low-reliability regions rather than showing a
confident-looking mask the model does not stand behind.

Run:
    python -m train.stage_b_decoder --epochs 24 --batch-size 12
    python -m train.stage_b_decoder --arch unet --epochs 2 --batch-size 2 --max-steps 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peri.core.canon import PERI_SEED, seed_everything, utc_now_iso
from peri.core.fragility import assert_transform_disjointness
from peri.core.videoprint import VideoprintExtractor
from train.config import (
    ARTIFACTS_DIR,
    CORPUS_DIR,
    STAGE_A_CKPT,
    STAGE_B,
    STAGE_B_CKPT,
    RunMeta,
)
from train.dataset import MaskDataset

assert_transform_disjointness()

IN_CHANNELS = 6
OUT_CHANNELS = 2


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class UNetDecoder(nn.Module):
    """Compact U-Net. The fallback architecture: no downloads, no transformers."""

    def __init__(self, in_channels: int = IN_CHANNELS, base: int = 32) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_channels, base)
        self.enc2 = block(base, base * 2)
        self.enc3 = block(base * 2, base * 4)
        self.enc4 = block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = block(base * 2, base)
        self.head = nn.Conv2d(base, OUT_CHANNELS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class SegformerDecoder(nn.Module):
    """SegFormer-B2 with its patch embedding widened to six input channels."""

    def __init__(self, backbone: str = STAGE_B.backbone) -> None:
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.net = SegformerForSemanticSegmentation.from_pretrained(
            backbone, num_labels=OUT_CHANNELS, ignore_mismatched_sizes=True
        )
        proj = self.net.segformer.encoder.patch_embeddings[0].proj
        widened = nn.Conv2d(
            IN_CHANNELS,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=proj.bias is not None,
        )
        with torch.no_grad():
            # Copy the pretrained RGB filters into the first three channels and
            # halve them, so the fingerprint channels start from a copy rather than
            # from noise and the pretrained response is not doubled.
            widened.weight[:, :3] = proj.weight * 0.5
            widened.weight[:, 3:] = proj.weight * 0.5
            if proj.bias is not None:
                widened.bias.copy_(proj.bias)
        self.net.segformer.encoder.patch_embeddings[0].proj = widened

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(arch: str, backbone: str) -> nn.Module:
    if arch == "segformer":
        try:
            return SegformerDecoder(backbone)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] SegFormer unavailable ({exc}); falling back to U-Net")
            return UNetDecoder()
    if arch == "unet":
        return UNetDecoder()
    raise ValueError(f"unknown arch: {arch!r}")


# ---------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    union = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def confidence_loss(
    mask_logits: torch.Tensor, conf_logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """TCP target: the model's own probability for the true class, detached."""
    with torch.no_grad():
        probability = torch.sigmoid(mask_logits)
        true_class_probability = target * probability + (1.0 - target) * (1.0 - probability)
    return F.mse_loss(torch.sigmoid(conf_logits), true_class_probability)


def iou_score(mask_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    predicted = (torch.sigmoid(mask_logits) > threshold).float()
    intersection = (predicted * target).sum().item()
    union = ((predicted + target) > 0).float().sum().item()
    return intersection / union if union > 0 else 1.0


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC. No sklearn dependency in the training loop."""
    if len(set(labels)) < 2:
        return float("nan")
    order = np.argsort(np.asarray(scores, dtype=float))
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    y = np.asarray(labels)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def frame_score(mask_logits: torch.Tensor, top_fraction: float = 0.02) -> torch.Tensor:
    """Frame-level anomaly score: the mean of the most suspicious pixels.

    A plain mean would drown a small spliced region in a large authentic frame.
    """
    probability = torch.sigmoid(mask_logits).flatten(1)
    k = max(1, int(probability.shape[1] * top_fraction))
    return probability.topk(k, dim=1).values.mean(dim=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_loader(args, split: str, shuffle: bool, augment_enabled: bool) -> DataLoader:
    dataset = MaskDataset(
        corpus_dir=args.corpus,
        split=split,
        crop_size=args.crop_size,
        seed=args.seed,
        augment_enabled=augment_enabled,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=args.workers > 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage B (tamper decoder).")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=STAGE_B_CKPT)
    parser.add_argument("--stage-a", type=Path, default=STAGE_A_CKPT)
    parser.add_argument("--arch", choices=("segformer", "unet"), default=STAGE_B.arch)
    parser.add_argument("--backbone", default=STAGE_B.backbone)
    parser.add_argument("--epochs", type=int, default=STAGE_B.epochs)
    parser.add_argument("--batch-size", type=int, default=STAGE_B.batch_size)
    parser.add_argument("--crop-size", type=int, default=STAGE_B.crop_size)
    parser.add_argument("--lr-encoder", type=float, default=STAGE_B.lr_encoder)
    parser.add_argument("--lr-decoder", type=float, default=STAGE_B.lr_decoder)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full epoch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=PERI_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    autocast_enabled = device.type == "cuda"

    fingerprint = VideoprintExtractor(
        checkpoint=args.stage_a if Path(args.stage_a).is_file() else None,
        device=str(device),
    )
    print(f"fingerprint source: {fingerprint.mode}")
    if fingerprint.mode == "srm-residual":
        print("  (Stage A checkpoint absent - training on the SRM placeholder)")

    train_loader = make_loader(args, "train", shuffle=True, augment_enabled=True)
    val_loader = make_loader(args, "val", shuffle=False, augment_enabled=False)

    model = build_model(args.arch, args.backbone).to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"Stage B: {type(model).__name__} params={n_parameters:,}")
    print(f"device={device}  batch={args.batch_size}  crop={args.crop_size}")

    encoder_parameters, decoder_parameters = [], []
    for name, parameter in model.named_parameters():
        (encoder_parameters if "encoder" in name else decoder_parameters).append(parameter)
    groups = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": args.lr_encoder})
    if decoder_parameters:
        groups.append({"params": decoder_parameters, "lr": args.lr_decoder})
    optimiser = torch.optim.AdamW(groups, weight_decay=STAGE_B.weight_decay)
    steps = args.max_steps or len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, args.epochs * steps)
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_iou = -1.0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, (image, mask, _label, _method) in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.no_grad():
                print_field = fingerprint.extract(image)
            stacked = torch.cat([image, print_field.to(image.dtype)], dim=1)

            with torch.autocast("cuda", torch.bfloat16, enabled=autocast_enabled):
                logits = model(stacked)
                mask_logits = logits[:, 0:1].float()
                conf_logits = logits[:, 1:2].float()
                loss = (
                    STAGE_B.bce_weight * F.binary_cross_entropy_with_logits(mask_logits, mask)
                    + STAGE_B.dice_weight * dice_loss(mask_logits, mask)
                    + STAGE_B.confidence_weight * confidence_loss(mask_logits, conf_logits, mask)
                )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            scheduler.step()

            running += float(loss.item()) * image.shape[0]
            seen += image.shape[0]

        train_loss = running / max(seen, 1)

        model.eval()
        ious, scores, labels = [], [], []
        with torch.no_grad():
            for step, (image, mask, label, _method) in enumerate(val_loader):
                if args.max_steps and step >= args.max_steps:
                    break
                image = image.to(device)
                mask = mask.to(device)
                stacked = torch.cat(
                    [image, fingerprint.extract(image).to(image.dtype)], dim=1
                )
                logits = model(stacked)
                mask_logits = logits[:, 0:1].float()
                ious.append(iou_score(mask_logits, mask))
                scores.extend(frame_score(mask_logits).cpu().tolist())
                labels.extend(label.tolist())

        mean_iou = float(np.mean(ious)) if ious else 0.0
        area = auroc(scores, labels)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_iou": mean_iou, "val_auroc": area}
        )
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
            f"IoU {mean_iou:.4f}  AUROC {area:.4f}"
        )

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "arch": type(model).__name__,
                        "requested_arch": args.arch,
                        "backbone": args.backbone,
                        "in_channels": IN_CHANNELS,
                        "out_channels": OUT_CHANNELS,
                        "crop_size": args.crop_size,
                    },
                    "meta": RunMeta(
                        stage="B",
                        extra={
                            "fingerprint_source": fingerprint.mode,
                            "best_val_iou": best_iou,
                            "val_auroc": area,
                            "epochs": args.epochs,
                            "finished_utc": utc_now_iso(),
                            "device": str(device),
                            "n_parameters": n_parameters,
                        },
                    ).to_dict(),
                    "history": history,
                },
                args.out,
            )
            print(f"  saved {args.out}")

    print(f"done in {(time.time() - started) / 60:.1f} min; best IoU {best_iou:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
