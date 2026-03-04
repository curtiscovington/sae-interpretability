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


def get_transformer_blocks(model: torch.nn.Module):
    """
    Return the canonical transformer block list for supported decoder-only families.

    Currently supports:
    - GPTNeoX (e.g., Pythia): model.gpt_neox.layers
    - Gemma/Gemma2 (HF): model.model.layers
    - Llama-style (HF): model.model.layers
    """
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(
        "Unsupported model architecture for block hooks. "
        "Expected GPTNeoX-style (model.gpt_neox.layers) or Llama/Gemma-style (model.model.layers)."
    )


def get_transformer_block(model: torch.nn.Module, layer_index: int):
    blocks = get_transformer_blocks(model)
    return blocks[layer_index]


def register_mlp_output_hook(model: torch.nn.Module, layer_index: int, hook_fn):
    """
    Register a forward hook on the block MLP output for supported model families.
    """
    block = get_transformer_block(model, layer_index)

    # GPTNeoX and many HF decoder families expose `.mlp`
    if hasattr(block, "mlp"):
        return block.mlp.register_forward_hook(hook_fn)

    raise ValueError(
        "Could not find `.mlp` module on selected transformer block; "
        "mlp_output hooks are unsupported for this architecture."
    )


@contextmanager
def activation_collector(model: torch.nn.Module, layer_index: int, stream: str = "mlp_output") -> Generator[list[torch.Tensor], None, None]:
    acts: list[torch.Tensor] = []

    def hook_mlp(_module, _inp, out):
        acts.append(out.detach())

    def hook_resid(_module, _inp, out):
        # hidden_states from full block output
        acts.append(out[0].detach() if isinstance(out, tuple) else out.detach())

    block = get_transformer_block(model, layer_index)
    if stream == "mlp_output":
        handle = register_mlp_output_hook(model, layer_index, hook_mlp)
    elif stream == "residual":
        handle = block.register_forward_hook(hook_resid)
    else:
        raise ValueError(f"Unsupported activation_stream: {stream}")

    try:
        yield acts
    finally:
        handle.remove()
