"""Stage A - self-supervised acquisition fingerprint (the Videoprint).

Trained on UNLABELLED AUTHENTIC VIDEO ONLY. It never sees a manipulated frame.
That is the point: the fingerprint learns what a consistent acquisition pipeline
looks like, so a region that came from somewhere else reads as a different
texture. A classifier trained on our own splices would only learn our splices.

Objective: NT-Xent (InfoNCE). Two patches drawn from the same clip at the same
GOP-position bucket are a positive pair; every other clip in the batch supplies
negatives. Patches from the same clip that are not the anchor's partner are masked
out of the negative set, since they are not true negatives.

Prior art credited on the report's Methods page: Noiseprint (2019), TruFor (CVPR
2023). The fingerprint paradigm is theirs; the video formulation is ours.

Run:
    python -m train.stage_a_videoprint --epochs 30 --batch-size 256
    python -m train.stage_a_videoprint --epochs 2 --batch-size 32 --pairs 2000   # smoke test
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peri.core.canon import PERI_SEED, seed_everything, utc_now_iso
from peri.core.fragility import assert_transform_disjointness
from peri.core.videoprint import DnCNN, ProjectionHead
from train.config import ARTIFACTS_DIR, CORPUS_DIR, STAGE_A, STAGE_A_CKPT, RunMeta
from train.dataset import ContrastivePatchDataset

assert_transform_disjointness()


def nt_xent(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    clip_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """NT-Xent over 2N embeddings, with same-clip non-partners masked out.

    Leaving same-clip patches in the negative set would ask the network to push
    apart two patches it is simultaneously being asked to pull together, and the
    fingerprint collapses.
    """
    batch = z_a.shape[0]
    embeddings = torch.cat([z_a, z_b], dim=0)
    ids = torch.cat([clip_ids, clip_ids], dim=0)

    similarity = embeddings @ embeddings.t() / temperature
    diagonal = torch.eye(2 * batch, dtype=torch.bool, device=embeddings.device)
    similarity = similarity.masked_fill(diagonal, float("-inf"))

    partner = torch.arange(2 * batch, device=embeddings.device)
    partner = (partner + batch) % (2 * batch)

    same_clip = ids[:, None] == ids[None, :]
    partner_mask = torch.zeros_like(same_clip)
    partner_mask[torch.arange(2 * batch, device=embeddings.device), partner] = True
    forbidden = same_clip & ~partner_mask & ~diagonal
    similarity = similarity.masked_fill(forbidden, float("-inf"))

    return F.cross_entropy(similarity, partner)


MIN_CLIPS_FOR_CONTRASTIVE = 2

# Fraction of total VRAM we assume is actually available for activations, after
# weights, optimiser state, the CUDA context and allocator fragmentation.
USABLE_VRAM_FRACTION = 0.80


def activation_bytes_per_sample(width: int, depth: int, patch: int, dtype_bytes: int) -> int:
    """Rough peak activation footprint for one patch through the DnCNN.

    Each Conv-BN-ReLU block keeps about two full-resolution feature maps alive
    for the backward pass: the convolution's input, and the batch norm's output
    (which ReLU then overwrites in place). There is no downsampling anywhere in
    this network, so every one of those maps is full patch resolution, which is
    what makes the footprint so large relative to the parameter count.
    """
    maps = 2 * (depth - 2) + 2
    return maps * width * patch * patch * dtype_bytes


def estimate_peak_bytes(batch: int, width: int, depth: int, patch: int, bf16: bool) -> int:
    """Peak activation bytes for one training step.

    Doubled because NT-Xent needs both views forward before either can be freed.
    """
    return 2 * batch * activation_bytes_per_sample(width, depth, patch, 2 if bf16 else 4)


def largest_batch_that_fits(
    budget_bytes: int, width: int, depth: int, patch: int, bf16: bool
) -> int:
    per_step = estimate_peak_bytes(1, width, depth, patch, bf16)
    return max(1, int(budget_bytes // max(per_step, 1)))


def check_memory_budget(args, device: torch.device) -> None:
    """Print the memory estimate, and stop before a run that cannot finish.

    Failing here with a number and a suggested batch size is far kinder than the
    allocator failing several minutes in with a message that does not say what
    to change.
    """
    bf16 = device.type == "cuda"
    needed = estimate_peak_bytes(
        args.batch_size, args.width, args.depth, STAGE_A.patch_size, bf16
    )

    if device.type != "cuda":
        print(
            f"activations: ~{needed / 1e9:.1f} GB of system RAM at batch "
            f"{args.batch_size} (CPU run, float32)"
        )
        return

    total = torch.cuda.get_device_properties(device).total_memory
    budget = int(total * USABLE_VRAM_FRACTION)
    print(
        f"VRAM: {total / 1e9:.1f} GB total, assuming {budget / 1e9:.1f} GB usable; "
        f"activations need ~{needed / 1e9:.1f} GB at batch {args.batch_size}"
    )
    if needed <= budget:
        return

    suggestion = largest_batch_that_fits(
        budget, args.width, args.depth, STAGE_A.patch_size, bf16
    )
    suggestion = max(8, 1 << (suggestion.bit_length() - 1))
    raise SystemExit(
        f"\nThis run needs about {needed / 1e9:.1f} GB of activation memory but "
        f"only about {budget / 1e9:.1f} GB is usable on this GPU, so it would "
        f"fail partway through.\n\n"
        f"Re-run with a smaller batch:\n\n"
        f"    python -m train.stage_a_videoprint --epochs {args.epochs} "
        f"--batch-size {suggestion}\n\n"
        f"CLAUDE.md section 4 specifies batch 256 against 12-16 GB of VRAM. A "
        f"smaller batch costs little here: NT-Xent draws its negatives from the "
        f"other clips in the batch, and this corpus has few enough source clips "
        f"that a larger batch mostly adds same-clip pairs, which are masked out "
        f"of the negative set anyway.\n"
        f"Pass --allow-oversized-batch to override this check."
    )


def build_loader(args, split: str, length: int) -> tuple[DataLoader, int]:
    """Return the loader and the number of distinct clips behind it."""
    dataset = ContrastivePatchDataset(
        corpus_dir=args.corpus,
        split=split,
        patch_size=STAGE_A.patch_size,
        gop_buckets=STAGE_A.gop_buckets,
        length=length,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # the dataset is already an infinite seeded sampler
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    return loader, dataset.n_clips


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage A (Videoprint).")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=STAGE_A_CKPT)
    parser.add_argument("--epochs", type=int, default=STAGE_A.epochs)
    parser.add_argument("--batch-size", type=int, default=STAGE_A.batch_size)
    parser.add_argument("--pairs", type=int, default=STAGE_A.pairs_per_epoch)
    parser.add_argument("--lr", type=float, default=STAGE_A.lr)
    parser.add_argument("--width", type=int, default=STAGE_A.width)
    parser.add_argument("--depth", type=int, default=STAGE_A.depth)
    parser.add_argument("--temperature", type=float, default=STAGE_A.temperature)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--allow-oversized-batch",
        action="store_true",
        help="skip the VRAM check and let the allocator decide",
    )
    parser.add_argument("--seed", type=int, default=PERI_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    autocast_enabled = device.type == "cuda"

    if not args.allow_oversized_batch:
        check_memory_budget(args, device)

    train_loader, n_train_clips = build_loader(args, "train", args.pairs)
    if n_train_clips < MIN_CLIPS_FOR_CONTRASTIVE:
        raise SystemExit(
            f"the train split has {n_train_clips} authentic clip(s). NT-Xent takes "
            f"its negatives from the other clips in the batch, so with fewer than "
            f"{MIN_CLIPS_FOR_CONTRASTIVE} there is nothing to push apart and the "
            f"loss is zero by construction. Add authentic source video to "
            f"data/authentic and rebuild the corpus."
        )
    if n_train_clips < args.batch_size:
        print(
            f"[warn] {n_train_clips} authentic clips vs batch size {args.batch_size}: "
            f"most in-batch pairs come from the same clip and are masked out of the "
            f"negative set, so each step sees far fewer effective negatives than the "
            f"batch size suggests. More source clips will help more than a bigger batch."
        )

    val_loader = None
    try:
        candidate, n_val_clips = build_loader(
            args, "val", max(args.pairs // 10, args.batch_size)
        )
    except Exception as exc:
        print(f"[warn] no validation split available ({exc}); selecting on train loss")
    else:
        if n_val_clips < MIN_CLIPS_FOR_CONTRASTIVE:
            # Every patch in the batch then comes from the same clip, every
            # negative is masked out as a same-clip non-partner, and the softmax
            # is left with a single unmasked column. Cross-entropy is exactly
            # 0.0 no matter what the weights are. Selecting the best epoch on
            # that number means selecting the first epoch, forever.
            print(
                f"[warn] the val split has {n_val_clips} authentic clip(s), which "
                f"makes the contrastive validation loss identically zero and "
                f"useless for checkpoint selection; selecting on train loss instead"
            )
        else:
            val_loader = candidate

    model = DnCNN(depth=args.depth, width=args.width).to(device)
    head = ProjectionHead(in_channels=3, out_dim=STAGE_A.embed_dim).to(device)
    parameters = list(model.parameters()) + list(head.parameters())
    print(
        f"Stage A: DnCNN depth={args.depth} width={args.width} "
        f"params={model.n_parameters():,} (+head {sum(p.numel() for p in head.parameters()):,})"
    )
    print(f"device={device}  batch={args.batch_size}  pairs/epoch={args.pairs}")

    optimiser = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=STAGE_A.weight_decay
    )
    steps_per_epoch = max(1, args.pairs // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs * steps_per_epoch
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = float("inf")
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        running, seen = 0.0, 0
        epoch_started = time.time()

        for patch_a, patch_b, clip_ids, _ in train_loader:
            patch_a = patch_a.to(device, non_blocking=True)
            patch_b = patch_b.to(device, non_blocking=True)
            clip_ids = clip_ids.to(device, non_blocking=True)

            with torch.autocast("cuda", torch.bfloat16, enabled=autocast_enabled):
                z_a = head(model(patch_a))
                z_b = head(model(patch_b))
                loss = nt_xent(z_a.float(), z_b.float(), clip_ids, args.temperature)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimiser.step()
            scheduler.step()

            running += float(loss.item()) * patch_a.shape[0]
            seen += patch_a.shape[0]

        train_loss = running / max(seen, 1)
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            head.eval()
            with torch.no_grad():
                total, count = 0.0, 0
                for patch_a, patch_b, clip_ids, _ in val_loader:
                    patch_a = patch_a.to(device)
                    patch_b = patch_b.to(device)
                    clip_ids = clip_ids.to(device)
                    z_a = head(model(patch_a))
                    z_b = head(model(patch_b))
                    total += float(
                        nt_xent(z_a.float(), z_b.float(), clip_ids, args.temperature).item()
                    ) * patch_a.shape[0]
                    count += patch_a.shape[0]
                val_loss = total / max(count, 1)

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        elapsed = time.time() - epoch_started
        print(
            f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}  {elapsed:.0f}s"
        )

        target = train_loss if math.isnan(val_loss) else val_loss
        if target < best:
            best = target
            torch.save(
                {
                    "model": model.state_dict(),
                    "head": head.state_dict(),
                    "config": {
                        "depth": args.depth,
                        "width": args.width,
                        "patch_size": STAGE_A.patch_size,
                        "embed_dim": STAGE_A.embed_dim,
                        "temperature": args.temperature,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                    },
                    "meta": RunMeta(
                        stage="A",
                        extra={
                            "trained_on": "authentic video only",
                            "objective": "NT-Xent over same-clip same-GOP-bucket patches",
                            "epochs": args.epochs,
                            "best_loss": best,
                            "finished_utc": utc_now_iso(),
                            "device": str(device),
                            "n_parameters": model.n_parameters(),
                        },
                    ).to_dict(),
                    "history": history,
                },
                args.out,
            )
            print(f"  saved {args.out}")

    print(f"done in {(time.time() - started) / 60:.1f} min; best loss {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
