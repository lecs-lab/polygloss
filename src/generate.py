import inspect
import itertools
import logging

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.config.experiment_config import ExperimentConfig
from src.distributed import DistributedParameters
from src.util.trim_and_left_pad import trim_and_left_pad

logger = logging.getLogger(__name__)


def generate(
    model,
    tokenizer,
    dataloader: DataLoader,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
) -> pd.DataFrame:
    """Runs inference, generating predictions for each item in the dataloader.

    Critically, the `original_dataset` should be the dataset *prior to tokenization*,
    which is used to get metadata about each predicted example.
    It must be the same length and order as the dataloader.

    Returns:
        (list[PredictedExample]): The generated outputs and decoded labels (or None if not provided)
    """
    model.eval()
    device = distributed_parameters["device"]

    generations: list[str] = []
    labels: list[str | None] = []
    task_keys: list[str] = []
    ids: list[str] = []

    if distributed_parameters["distributed"]:
        model = model.module
        dataloader.sampler.set_epoch(0)  # type:ignore

    for batch in (
        tqdm(dataloader, desc="Generating")
        if distributed_parameters["rank"] == 0
        else dataloader
    ):
        if config.model_type == "decoder":
            batch = trim_and_left_pad(batch, tokenizer.pad_token_id)

        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in inspect.signature(model.forward).parameters
        }

        batch_generations = model.generate(
            **inputs,
            do_sample=False,
            num_beams=config.num_beams,
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if config.model_type == "decoder":
            batch_generations = batch_generations[:, inputs["input_ids"].size(-1) :]
        generations.extend(
            tokenizer.batch_decode(batch_generations, skip_special_tokens=True)
        )

        if "labels" in batch:
            batch["labels"][batch["labels"] == -100] = tokenizer.pad_token_id
            labels.extend(
                tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            )
        else:
            labels.extend([None] * len(batch_generations))
        task_keys.extend(batch["task"])
        ids.extend(batch["id"])

    # Gather all examples
    if distributed_parameters["distributed"]:
        all_generations = [None for _ in range(distributed_parameters["world_size"])]
        all_labels = [None for _ in range(distributed_parameters["world_size"])]
        all_task_keys = [None for _ in range(distributed_parameters["world_size"])]
        all_ids = [None for _ in range(distributed_parameters["world_size"])]

        logger.info(
            f"[RANK {distributed_parameters['rank']}] Finished generation, entering gather"
        )
        torch.distributed.all_gather_object(all_generations, generations)
        torch.distributed.all_gather_object(all_labels, labels)
        torch.distributed.all_gather_object(all_task_keys, task_keys)
        torch.distributed.all_gather_object(all_ids, ids)

        # all_gather creates a list of lists, so we need to flatten
        all_generations = list(itertools.chain.from_iterable(all_generations))  # type:ignore
        all_labels = list(itertools.chain.from_iterable(all_labels))  # type:ignore
        all_task_keys = list(
            itertools.chain.from_iterable(all_task_keys)  # type:ignore
        )
        all_ids = list(itertools.chain.from_iterable(all_ids))  # type:ignore
    else:
        all_generations = generations
        all_labels = labels
        all_task_keys = task_keys
        all_ids = ids
    assert all(gen is not None for gen in all_generations)
    df = pd.DataFrame(
        [
            {
                "predicted": gen,
                "reference": label,
                "task": task_key,
                "id": id,
            }
            for gen, label, task_key, id in zip(
                all_generations, all_labels, all_task_keys, all_ids
            )
        ]
    )
    # De-dupe, since with DDP we might have dupe generations due to weird batch sizes
    df = df.drop_duplicates(subset=["id", "task"], keep="first").reset_index(drop=True)
    return df
