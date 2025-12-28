import torch


def trim_and_left_pad(batch, pad_token_id: int):
    """Given a batch with `input_ids`, `prompt_lengths` and `attention_mask`,
    trims each sequence to its `prompt_length`, and then re-pads on the left."""
    assert all(k in batch for k in ["input_ids", "prompt_lengths", "attention_mask"])

    device = batch["input_ids"].device
    batch_size = batch["input_ids"].size(0)
    max_length = batch["prompt_lengths"].max()
    input_ids = torch.full((batch_size, max_length), pad_token_id, device=device)
    attention_mask = torch.full((batch_size, max_length), 0, device=device)

    for idx in range(batch_size):
        prompt_length = batch["prompt_lengths"][idx]
        input_ids[idx][max_length - prompt_length :] = batch["input_ids"][idx][
            :prompt_length
        ]
        attention_mask[idx][max_length - prompt_length :] = batch["attention_mask"][
            idx
        ][:prompt_length]
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    return batch
