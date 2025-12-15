from typing import Any, Dict, List

from transformers.data.data_collator import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling


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


class FlexibleCausalLMCollator(DataCollatorForLanguageModeling):
    """
    This mirrors the behavior of FlexibleSeq2SeqCollator but for causal LMs.
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
                    f.pop(k, None)

        batch = super().__call__(features)
        
        # Add the untouched extras back
        batch.update(extras)
        return batch
