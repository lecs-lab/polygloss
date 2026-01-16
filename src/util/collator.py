from typing import Any, Dict, List

import torch
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)


def _is_tensor_like(v: Any) -> bool:
    """
    True if v is a list/array of ints/floats suitable for tokenizer.pad().
    We only test the first element to keep it fast.
    """
    if not isinstance(v, list) or len(v) == 0:
        return False
    first = v[0]
    # If it’s already numeric (int/float) or a list of numerics, pad can handle it
    if isinstance(first, (int, float)):
        return True
    if isinstance(first, list):
        # nested list of ints/floats is fine (token ids etc.)
        return all(isinstance(x, (int, float)) for x in first)
    return False


class FlexibleSeq2SeqCollator(DataCollatorForSeq2Seq):
    """
    Like DataCollatorForSeq2Seq, but automatically carries through any fields
    whose values are not tensor-like (e.g. raw text, IDs, metadata).
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Collect names of fields to preserve
        if not features:
            return {}
        keys = list(features[0].keys())

        # Extract non-tensor fields
        extras: Dict[str, List[Any]] = {}
        for k in keys:
            values = [f[k] for f in features if k in f]
            if values and not _is_tensor_like(values):
                extras[k] = values
                for f in features:
                    f.pop(k, None)  # remove before parent collator

        # Let DataCollatorForSeq2Seq do its usual padding on the remaining numeric fields
        batch = super().__call__(features)

        # Add the untouched extras back
        batch.update(extras)
        return batch


class FlexibleCollatorWithPadding(DataCollatorForLanguageModeling):
    def __init__(self, label_pad_token_id: int, *args, **kwargs):
        super(FlexibleCollatorWithPadding, self).__init__(*args, **kwargs)
        self.label_pad_token_id = label_pad_token_id

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors: str | None = None
    ) -> Dict[str, Any]:
        # Collect names of fields to preserve
        if not features:
            return {}
        keys = list(features[0].keys())

        # Extract non-tensor fields
        extras: Dict[str, List[Any]] = {}
        for k in keys:
            values = [f[k] for f in features if k in f]
            if values and not _is_tensor_like(values):
                extras[k] = values
                for f in features:
                    f.pop(k, None)  # remove before parent collator

        # Let DataCollatorForLanguageModeling do its usual padding on the remaining numeric fields
        batch = super().__call__(features, return_tensors)

        # Mask labels for prompt
        prompt_mask = torch.arange(batch["input_ids"].size(-1)).expand(
            batch["input_ids"].shape
        )
        prompt_mask = prompt_mask > (
            batch["input_ids"].size(-1) - batch["label_lengths"]
        ).unsqueeze(-1)
        batch["labels"].masked_fill_(prompt_mask, -100)

        # Add the untouched extras back
        batch.update(extras)
        return batch
