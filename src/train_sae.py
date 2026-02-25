from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import load_config
from .sae import SparseAutoencoder, sae_loss
from .utils import get_device, set_seed


def _train_one(cfg, label: str) -> dict:
    acts_dir = Path(cfg.collection.output_dir)
    ckpt_dir = Path(cfg.outputs.checkpoints_dir)
    table_dir = Path(cfg.outputs.tables_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    with open(acts_dir / f"meta_{label}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = int(meta["tokens_collected"])
    d_model = int(meta["d_model"])
    arr = np.memmap(meta["acts_path"], mode="r", dtype=np.float16, shape=(cfg.collection.__dict__[f'tokens_{label.lower()}'], d_model))
    x = np.array(arr[:n], dtype=np.float32)

    split = int(0.9 * n)
    x_train = torch.from_numpy(x[:split])
    x_val = torch.from_numpy(x[split:])

    train_loader = DataLoader(TensorDataset(x_train), batch_size=cfg.sae.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=cfg.sae.batch_size, shuffle=False)

    device = get_device(cfg.device_preference)
    model = SparseAutoencoder(
        d_model=d_model,
        d_sae=cfg.sae.d_sae,
        sparsity_mode=cfg.sae.sparsity_mode,
        topk=cfg.sae.topk,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.sae.lr,
        weight_decay=cfg.sae.weight_decay,
    )
    total_steps = cfg.sae.epochs * max(len(train_loader), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(total_steps, 1))

    logs = []
    step = 0
    for epoch in range(cfg.sae.epochs):
        model.train()
        train_recon = 0.0
        train_l1 = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"train {label} e{epoch+1}/{cfg.sae.epochs}")
        for (xb,) in pbar:
            xb = xb.to(device)
            recon, h = model(xb)
            loss_out = sae_loss(xb, recon, h, cfg.sae.l1_coeff)

            optim.zero_grad(set_to_none=True)
            loss_out.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.sae.grad_clip)
            optim.step()
            sched.step()

            train_recon += float(loss_out.recon.item()) * xb.shape[0]
            train_l1 += float(loss_out.l1.item()) * xb.shape[0]
            count += xb.shape[0]
            step += 1
            if step % cfg.sae.checkpoint_every == 0:
                torch.save(model.state_dict(), ckpt_dir / f"sae_{label}_step{step}.pt")

        model.eval()
        val_recon = 0.0
        val_l1 = 0.0
        vcount = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, h = model(xb)
                out = sae_loss(xb, recon, h, cfg.sae.l1_coeff)
                val_recon += float(out.recon.item()) * xb.shape[0]
                val_l1 += float(out.l1.item()) * xb.shape[0]
                vcount += xb.shape[0]

        row = {
            "label": label,
            "epoch": epoch + 1,
            "train_recon": train_recon / max(count, 1),
            "train_l1": train_l1 / max(count, 1),
            "val_recon": val_recon / max(vcount, 1),
            "val_l1": val_l1 / max(vcount, 1),
            "lr": sched.get_last_lr()[0],
        }
        logs.append(row)
        print(row)

    final_ckpt = ckpt_dir / f"sae_{label}.pt"
    torch.save(model.state_dict(), final_ckpt)

    log_path = table_dir / f"train_log_{label}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)

    return {
        "label": label,
        "checkpoint": str(final_ckpt),
        "train_log_csv": str(log_path),
        "d_model": d_model,
        "d_sae": cfg.sae.d_sae,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    out_a = _train_one(cfg, "A")
    out_b = _train_one(cfg, "B")

    with open(Path(cfg.outputs.root) / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump({"A": out_a, "B": out_b}, f, indent=2)


if __name__ == "__main__":
    main()
