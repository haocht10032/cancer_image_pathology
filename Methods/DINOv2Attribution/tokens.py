"""Patch-token extraction for DINOv2's native 16 x 16 grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import DINO_NATIVE_GRID_SIZE, DINOv2Classifier


@torch.inference_mode()
def cache_dinov2_patch_tokens(
    model: DINOv2Classifier,
    loader: DataLoader,
    token_path: Path | str,
    index_path: Path | str,
    device: torch.device,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Stream DINOv2 patch tokens to a float16 NumPy array."""
    token_path = Path(token_path)
    index_path = Path(index_path)
    if token_path.is_file() and index_path.is_file() and not overwrite:
        return token_path, index_path

    token_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = len(loader.dataset)
    token_store = np.lib.format.open_memmap(
        token_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            sample_count,
            DINO_NATIVE_GRID_SIZE,
            DINO_NATIVE_GRID_SIZE,
            model.feature_dim,
        ),
    )
    index_rows: list[dict[str, object]] = []
    offset = 0
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        tokens = model.forward_tokens(images)
        patch_tokens = tokens[:, 1:]
        expected = DINO_NATIVE_GRID_SIZE**2
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(
                f"Expected {expected} DINOv2 patch tokens, "
                f"found {patch_tokens.shape[1]}"
            )
        patch_tokens = patch_tokens.reshape(
            len(images),
            DINO_NATIVE_GRID_SIZE,
            DINO_NATIVE_GRID_SIZE,
            model.feature_dim,
        )
        batch_size = len(images)
        token_store[offset : offset + batch_size] = (
            patch_tokens.detach().cpu().numpy().astype(np.float16)
        )
        for position in range(batch_size):
            index_rows.append(
                {
                    "token_index": offset + position,
                    "path": batch["path"][position],
                    "case_id": batch["case_id"][position],
                    "label": int(batch["label"][position]),
                }
            )
        offset += batch_size
    token_store.flush()
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    return token_path, index_path
