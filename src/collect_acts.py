from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import load_config
from .data import TextStreamSpec, load_text_stream, token_batches
from .model import activation_collector, load_model_and_tokenizer
from .utils import get_device, set_seed


def _collect_one_dataset(cfg, label: str, spec: TextStreamSpec, tokens_target: int) -> dict:
    out_dir = Path(cfg.collection.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg.device_preference)
    hooked = load_model_and_tokenizer(cfg.model.model_name, cfg.model.dtype, device)

    d_model = int(hooked.model.config.hidden_size)
    acts_path = out_dir / f"acts_{label}.mmap"
    toks_path = out_dir / f"tokens_{label}.npy"

    acts_mmap = np.memmap(acts_path, mode="w+", dtype=np.float16, shape=(tokens_target, d_model))
    token_ids = np.zeros((tokens_target,), dtype=np.int32)

    text_iter = load_text_stream(spec)
    batches = token_batches(
        text_iter,
        tokenizer=hooked.tokenizer,
        seq_len=cfg.collection.seq_len,
        batch_size=cfg.collection.batch_size,
        total_tokens_target=tokens_target,
    )

    idx = 0
    t0 = time.time()

    with torch.inference_mode(), activation_collector(
        hooked.model, cfg.model.layer_index, cfg.model.activation_stream
    ) as acts:
        pbar = tqdm(desc=f"collect:{label}", total=tokens_target)
        for input_ids, attention_mask, _ in batches:
            if idx >= tokens_target:
                break
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            _ = hooked.model(input_ids=input_ids, attention_mask=attention_mask)

            layer_act = acts.pop()
            layer_act = layer_act.float().cpu().numpy().reshape(-1, d_model)
            toks = input_ids.detach().cpu().numpy().reshape(-1)

            n = min(layer_act.shape[0], tokens_target - idx)
            acts_mmap[idx : idx + n] = layer_act[:n].astype(np.float16)
            token_ids[idx : idx + n] = toks[:n]
            idx += n
            pbar.update(n)
        pbar.close()

    acts_mmap.flush()
    np.save(toks_path, token_ids[:idx])

    elapsed = time.time() - t0
    mb = (idx * d_model * 2) / (1024 * 1024)
    meta = {
        "label": label,
        "tokens_collected": idx,
        "d_model": d_model,
        "acts_path": str(acts_path),
        "tokens_path": str(toks_path),
        "dtype": "float16",
        "throughput_tokens_per_sec": idx / max(elapsed, 1e-6),
        "storage_mb": mb,
        "model_name": cfg.model.model_name,
        "layer_index": cfg.model.layer_index,
        "activation_stream": cfg.model.activation_stream,
    }
    with open(out_dir / f"meta_{label}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{label}] tokens={idx} throughput={meta['throughput_tokens_per_sec']:.1f}/s storage={mb:.1f}MB")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    spec_a = TextStreamSpec(
        name=cfg.data.dataset_a_name,
        config=cfg.data.dataset_a_config,
        split=cfg.data.dataset_a_split,
        text_field=cfg.data.text_field_a,
        max_chars_per_example=cfg.data.max_chars_per_example,
        cache_dir=cfg.data.cache_dir,
    )
    spec_b = TextStreamSpec(
        name=cfg.data.dataset_b_name,
        config=cfg.data.dataset_b_config,
        split=cfg.data.dataset_b_split,
        text_field=cfg.data.text_field_b,
        max_chars_per_example=cfg.data.max_chars_per_example,
        cache_dir=cfg.data.cache_dir,
    )

    meta_a = _collect_one_dataset(cfg, "A", spec_a, cfg.collection.tokens_a)
    meta_b = _collect_one_dataset(cfg, "B", spec_b, cfg.collection.tokens_b)

    with open(Path(cfg.collection.output_dir) / "meta_all.json", "w", encoding="utf-8") as f:
        json.dump({"A": meta_a, "B": meta_b}, f, indent=2)


if __name__ == "__main__":
    main()
