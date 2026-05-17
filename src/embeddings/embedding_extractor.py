from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def load_codebert() -> Tuple[RobertaTokenizer, RobertaModel, str]:
    device = get_device()
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


def get_codebert_embedding(code: str) -> np.ndarray:
    code = str(code or "").strip()
    if not code:
        raise ValueError("Code input is empty.")

    tokenizer, model, device = load_codebert()

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0).cpu().numpy().astype(np.float32)
