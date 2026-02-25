from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Generator, Iterable

import requests


WIKI_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
CODE_URLS = [
    "https://raw.githubusercontent.com/python/cpython/main/Lib/ast.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/tokenize.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/pathlib.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/re.py",
    "https://raw.githubusercontent.com/python/cpython/main/Lib/json/decoder.py",
    "https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py",
    "https://raw.githubusercontent.com/pallets/flask/main/src/flask/cli.py",
    "https://raw.githubusercontent.com/numpy/numpy/main/numpy/linalg/__init__.py",
    "https://raw.githubusercontent.com/numpy/numpy/main/numpy/core/fromnumeric.py",
]


@dataclass
class TextStreamSpec:
    name: str
    config: str | None
    split: str
    text_field: str
    max_chars_per_example: int
    cache_dir: str


def _download_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _chunk_nonempty(text: str) -> list[str]:
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def load_text_stream(spec: TextStreamSpec) -> Iterable[str]:
    name = spec.name.lower()

    if "wiki" in name:
        text = _download_text(WIKI_URL)
        chunks = [ln.strip() for ln in text.splitlines() if ln.strip()]
        while True:
            for c in chunks:
                yield c[: spec.max_chars_per_example]

    if "code" in name or "github" in name:
        corpora: list[str] = []
        for u in CODE_URLS:
            try:
                corpora.extend(_chunk_nonempty(_download_text(u)))
            except requests.RequestException as e:
                print(f"[warn] skipping code source {u}: {e}")
                continue

        if not corpora:
            raise RuntimeError("No code sources could be downloaded.")

        while True:
            for c in corpora:
                yield c[: spec.max_chars_per_example]

    raise ValueError(f"Unsupported dataset name: {spec.name}. Use wiki-like for A and code-like for B.")


def token_batches(
    texts: Iterable[str],
    tokenizer,
    seq_len: int,
    batch_size: int,
    total_tokens_target: int,
) -> Generator[tuple, None, None]:
    """Yield (input_ids, attention_mask, token_texts) as fixed-size batches."""
    buffer: list[int] = []
    token_texts: list[str] = []
    produced = 0

    for txt in texts:
        ids = tokenizer.encode(txt, add_special_tokens=False)
        if not ids:
            continue
        buffer.extend(ids)
        token_texts.append(txt)

        while len(buffer) >= seq_len * batch_size and produced < total_tokens_target:
            chunk = list(islice(buffer, 0, seq_len * batch_size))
            del buffer[: seq_len * batch_size]
            import torch

            x = torch.tensor(chunk, dtype=torch.long).reshape(batch_size, seq_len)
            attn = torch.ones_like(x)
            produced += x.numel()
            yield x, attn, token_texts[-batch_size:]
            if produced >= total_tokens_target:
                return
