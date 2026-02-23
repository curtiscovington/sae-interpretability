from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    dataset_a_name: str
    dataset_a_config: Optional[str]
    dataset_a_split: str
    dataset_b_name: str
    dataset_b_config: Optional[str]
    dataset_b_split: str
    text_field_a: str
    text_field_b: str
    max_chars_per_example: int
    cache_dir: str


@dataclass
class ModelConfig:
    model_name: str
    layer_index: int
    activation_stream: str
    dtype: str


@dataclass
class CollectionConfig:
    seq_len: int
    batch_size: int
    num_workers: int
    tokens_a: int
    tokens_b: int
    chunk_size: int
    output_dir: str


@dataclass
class SAEConfig:
    d_sae: int
    lr: float
    batch_size: int
    epochs: int
    l1_coeff: float
    grad_clip: float
    checkpoint_every: int
    weight_decay: float
    scheduler: str
    recon_loss: str


@dataclass
class InterpretConfig:
    top_features: int
    top_contexts: int
    context_window_tokens: int


@dataclass
class OutputConfig:
    root: str
    results_json: str
    figures_dir: str
    tables_dir: str
    features_dir: str
    checkpoints_dir: str


@dataclass
class ExperimentConfig:
    seed: int
    device_preference: str
    data: DataConfig
    model: ModelConfig
    collection: CollectionConfig
    sae: SAEConfig
    interpret: InterpretConfig
    outputs: OutputConfig


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = _load_yaml(path)
    return ExperimentConfig(
        seed=raw["seed"],
        device_preference=raw["device_preference"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        collection=CollectionConfig(**raw["collection"]),
        sae=SAEConfig(**raw["sae"]),
        interpret=InterpretConfig(**raw["interpret"]),
        outputs=OutputConfig(**raw["outputs"]),
    )
