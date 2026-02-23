from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HookedModel:
    tokenizer: any
    model: torch.nn.Module


def load_model_and_tokenizer(model_name: str, dtype: str = "float16", device: torch.device | None = None) -> HookedModel:
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model.eval()
    if device is not None:
        model.to(device)

    return HookedModel(tokenizer=tokenizer, model=model)


@contextmanager
def activation_collector(model: torch.nn.Module, layer_index: int, stream: str = "mlp_output") -> Generator[list[torch.Tensor], None, None]:
    acts: list[torch.Tensor] = []

    def hook_mlp(_module, _inp, out):
        acts.append(out.detach())

    def hook_resid(_module, _inp, out):
        # hidden_states from full block output
        acts.append(out[0].detach() if isinstance(out, tuple) else out.detach())

    if not hasattr(model, "gpt_neox"):
        raise ValueError("This project currently expects GPTNeoX-style models (e.g., pythia).")

    block = model.gpt_neox.layers[layer_index]
    if stream == "mlp_output":
        handle = block.mlp.register_forward_hook(hook_mlp)
    elif stream == "residual":
        handle = block.register_forward_hook(hook_resid)
    else:
        raise ValueError(f"Unsupported activation_stream: {stream}")

    try:
        yield acts
    finally:
        handle.remove()
