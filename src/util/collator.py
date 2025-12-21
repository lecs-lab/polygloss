import torch
from typing import Any, Dict, List

from transformers.data.data_collator import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import BatchEncoding

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
    """Fixed version with explicit padding"""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}
        
        keys = list(features[0].keys())
        
        extras: Dict[str, List[Any]] = {}
        for k in keys:
            values = [f[k] for f in features if k in f]
            if values and not _is_tensor_like(values):
                extras[k] = values
                for f in features:
                    f.pop(k, None)
        
        batch = self._pad_batch(features)
        batch = BatchEncoding(batch)
        
        batch.update(extras)
        
        return batch
    
    def _pad_batch(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Explicitly pad input_ids, attention_mask, and labels"""
        
        # Find max length in this batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
        }
        
        has_labels = "labels" in features[0]
        if has_labels:
            batch["labels"] = []
        
        for feature in features:
            input_ids = feature["input_ids"]
            seq_length = len(input_ids)
            padding_length = max_length - seq_length
            
            # Pad input_ids with pad_token_id
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            
            # Create attention_mask (1 for real, 0 for padding)
            if "attention_mask" in feature:
                attention_mask = feature["attention_mask"] + [0] * padding_length
            else:
                attention_mask = [1] * seq_length + [0] * padding_length
            
            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(attention_mask)
            
            # Pad labels with -100 (ignore index)
            if has_labels:
                labels = feature["labels"]
                padded_labels = labels + [-100] * padding_length
                batch["labels"].append(padded_labels)
        
        # Convert to tensors
        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        
        return batch
