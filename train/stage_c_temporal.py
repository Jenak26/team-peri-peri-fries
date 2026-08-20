"""Stage C - temporal transformer over cached per-frame tokens.

Two steps, both driven from this file:

    python -m train.stage_c_temporal --cache      # run Stage B over the corpus once
    python -m train.stage_c_temporal --epochs 60  # train on the cached tokens

Per-frame token (8 dimensions), all derived from Stage B outputs and the
fingerprint field:

    0  top-2% mean tamper probability      4  mean reliability
    1  mean tamper probability             5  minimum reliability
    2  maximum tamper probability          6  fingerprint mean absolute response
    3  thresholded mask area fraction      7  fingerprint standard deviation

The model emits a video-level verdict and a per-frame tamper logit; the per-frame
head is the tamper timeline the frontend draws.

Partial-duration sequences: a corpus where every manipulated clip is manipulated in
every frame teaches a timeline nothing. At cache time we therefore also build hybrid
sequences by splicing the authentic and manipulated token streams of the SAME source
identity, so a run of tampered frames sits inside an otherwise authentic sequence.
The frame labels of those hybrids are exact by construction. This is documented in
the report rather than presented as if the corpus contained natural partial edits.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from peri.core.canon import PERI_SEED, seed_everything, utc_now_iso
from peri.core.fragility import assert_transform_disjointness
from peri.core.videoprint import VideoprintExtractor
from train.config import (
    ARTIFACTS_DIR,
    CORPUS_DIR,
    STAGE_A_CKPT,
    STAGE_B_CKPT,
    STAGE_C,
    STAGE_C_CKPT,
    RunMeta,
)
from train.dataset import _read_rgb, _frame_paths, load_index
from train.stage_b_decoder import build_model, frame_score

assert_transform_disjointness()

TOKEN_DIM = 8
TOKENS_PATH = CORPUS_DIR / "tokens_stage_c.pt"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@torch.no_grad()
def cache_tokens(
    corpus_dir: Path,
    stage_a: Path,
    stage_b: Path,
    device: torch.device,
    out_path: Path,
    max_frames: int,
) -> dict:
    if not stage_b.is_file():
        raise FileNotFoundError(
            f"Stage B checkpoint not found at {stage_b}. Train Stage B before caching."
        )

    payload = torch.load(stage_b, map_location=device, weights_only=False)
    config = payload.get("config", {})
    model = build_model(
        "unet" if config.get("arch") == "UNetDecoder" else "segformer",
        config.get("backbone", "nvidia/mit-b2"),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    fingerprint = VideoprintExtractor(
        checkpoint=stage_a if stage_a.is_file() else None, device=str(device)
    )
    index = load_index(corpus_dir)
    cached: dict[str, dict] = {}

    for position, sample in enumerate(index["samples"]):
        paths = _frame_paths(corpus_dir, sample)[:max_frames]
        if not paths:
            continue
        tokens = []
        for path in paths:
            image = _read_rgb(path)
            tensor = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )[None].to(device)
            field = fingerprint.extract(tensor).to(tensor.dtype)
            logits = model(torch.cat([tensor, field], dim=1))
            mask_logits = logits[:, 0:1].float()
            conf = torch.sigmoid(logits[:, 1:2].float())
            probability = torch.sigmoid(mask_logits)
            tokens.append(
                [
                    float(frame_score(mask_logits).item()),
                    float(probability.mean().item()),
                    float(probability.max().item()),
                    float((probability > 0.5).float().mean().item()),
                    float(conf.mean().item()),
                    float(conf.min().item()),
                    float(field.abs().mean().item()),
                    float(field.std().item()),
                ]
            )
        cached[sample["sample_id"]] = {
            "tokens": np.asarray(tokens, dtype=np.float32),
            "split": sample["split"],
            "label": int(sample["label"]),
            "method": sample["method"],
            "identity": sample["source_identity"],
        }
        if (position + 1) % 20 == 0:
            print(f"  cached {position + 1}/{len(index['samples'])}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": cached, "token_dim": TOKEN_DIM}, out_path)
    print(f"wrote {out_path} ({len(cached)} samples)")
    return cached


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TokenSequenceDataset(Dataset):
    def __init__(
        self,
        tokens_path: Path = TOKENS_PATH,
        split: str = "train",
        max_frames: int = STAGE_C.max_frames,
        hybrids: bool = True,
        seed: int = PERI_SEED,
    ) -> None:
        payload = torch.load(tokens_path, map_location="cpu", weights_only=False)
        cached: dict[str, dict] = payload["tokens"]
        self.max_frames = max_frames
        self.items: list[tuple[np.ndarray, np.ndarray, int]] = []

        by_identity: dict[str, dict[str, dict]] = {}
        for sample_id, entry in cached.items():
            if entry["split"] != split:
                continue
            tokens = entry["tokens"]
            frame_labels = np.full(len(tokens), entry["label"], dtype=np.float32)
            self.items.append((tokens, frame_labels, entry["label"]))
            by_identity.setdefault(entry["identity"], {})[
                "authentic" if entry["label"] == 0 else sample_id
            ] = entry

        if hybrids:
            rng = np.random.default_rng(seed)
            for identity, group in sorted(by_identity.items()):
                authentic = group.get("authentic")
                manipulated = [v for k, v in sorted(group.items()) if k != "authentic"]
                if authentic is None or not manipulated:
                    continue
                for entry in manipulated:
                    length = min(len(authentic["tokens"]), len(entry["tokens"]))
                    if length < 4:
                        continue
                    start = int(rng.integers(0, max(1, length // 2)))
                    end = int(rng.integers(start + 2, length + 1))
                    tokens = authentic["tokens"][:length].copy()
                    tokens[start:end] = entry["tokens"][start:end]
                    frame_labels = np.zeros(length, dtype=np.float32)
                    frame_labels[start:end] = 1.0
                    self.items.append((tokens, frame_labels, 1))

        if not self.items:
            raise RuntimeError(f"no cached token sequences for split {split!r}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int):
        tokens, frame_labels, label = self.items[item]
        length = min(len(tokens), self.max_frames)
        padded = np.zeros((self.max_frames, TOKEN_DIM), dtype=np.float32)
        labels = np.zeros(self.max_frames, dtype=np.float32)
        valid = np.zeros(self.max_frames, dtype=np.float32)
        padded[:length] = tokens[:length]
        labels[:length] = frame_labels[:length]
        valid[:length] = 1.0
        return (
            torch.from_numpy(padded),
            torch.from_numpy(labels),
            torch.from_numpy(valid),
            torch.tensor(float(label)),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TemporalTransformer(nn.Module):
    def __init__(self, config=STAGE_C) -> None:
        super().__init__()
        self.input = nn.Linear(TOKEN_DIM, config.d_model)
        self.position = nn.Parameter(torch.zeros(1, config.max_frames, config.d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.frame_head = nn.Linear(config.d_model, 1)
        self.video_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor):
        hidden = self.input(tokens) + self.position[:, : tokens.shape[1]]
        hidden = self.encoder(hidden, src_key_padding_mask=(valid < 0.5))
        frame_logits = self.frame_head(hidden).squeeze(-1)
        weights = valid / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return frame_logits, self.video_head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage C (temporal).")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--tokens", type=Path, default=TOKENS_PATH)
    parser.add_argument("--out", type=Path, default=STAGE_C_CKPT)
    parser.add_argument("--stage-a", type=Path, default=STAGE_A_CKPT)
    parser.add_argument("--stage-b", type=Path, default=STAGE_B_CKPT)
    parser.add_argument("--cache", action="store_true", help="cache tokens and exit")
    parser.add_argument("--epochs", type=int, default=STAGE_C.epochs)
    parser.add_argument("--batch-size", type=int, default=STAGE_C.batch_size)
    parser.add_argument("--lr", type=float, default=STAGE_C.lr)
    parser.add_argument("--seed", type=int, default=PERI_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    if args.cache:
        cache_tokens(
            args.corpus, args.stage_a, args.stage_b, device, args.tokens,
            STAGE_C.max_frames,
        )
        return 0

    if not args.tokens.is_file():
        print(f"tokens not found at {args.tokens}; run with --cache first")
        return 2

    train_loader = DataLoader(
        TokenSequenceDataset(args.tokens, "train", seed=args.seed),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TokenSequenceDataset(args.tokens, "val", hybrids=False, seed=args.seed),
        batch_size=args.batch_size, shuffle=False,
    )

    model = TemporalTransformer().to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"Stage C: TemporalTransformer params={n_parameters:,}  device={device}")

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=STAGE_C.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, args.epochs * len(train_loader))
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = -1.0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for tokens, frame_labels, valid, video_label in train_loader:
            tokens = tokens.to(device)
            frame_labels = frame_labels.to(device)
            valid = valid.to(device)
            video_label = video_label.to(device)

            frame_logits, video_logit = model(tokens, valid)
            frame_loss = (
                F.binary_cross_entropy_with_logits(
                    frame_logits, frame_labels, reduction="none"
                )
                * valid
            ).sum() / valid.sum().clamp(min=1.0)
            video_loss = F.binary_cross_entropy_with_logits(video_logit, video_label)
            loss = video_loss + 0.5 * frame_loss

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            scheduler.step()

            running += float(loss.item()) * tokens.shape[0]
            seen += tokens.shape[0]

        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for tokens, _frame_labels, valid, video_label in val_loader:
                _, video_logit = model(tokens.to(device), valid.to(device))
                scores.extend(torch.sigmoid(video_logit).cpu().tolist())
                labels.extend(video_label.int().tolist())

        from train.stage_b_decoder import auroc

        area = auroc(scores, labels)
        train_loss = running / max(seen, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auroc": area})
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  AUROC {area:.4f}")

        if area == area and area > best:
            best = area
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "d_model": STAGE_C.d_model,
                        "n_heads": STAGE_C.n_heads,
                        "n_layers": STAGE_C.n_layers,
                        "max_frames": STAGE_C.max_frames,
                        "token_dim": TOKEN_DIM,
                    },
                    "meta": RunMeta(
                        stage="C",
                        extra={
                            "best_val_auroc": best,
                            "epochs": args.epochs,
                            "finished_utc": utc_now_iso(),
                            "device": str(device),
                            "n_parameters": n_parameters,
                            "hybrid_sequences": True,
                        },
                    ).to_dict(),
                    "history": history,
                },
                args.out,
            )
            print(f"  saved {args.out}")

    print(f"done in {(time.time() - started) / 60:.1f} min; best AUROC {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
