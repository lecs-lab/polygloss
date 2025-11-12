import inspect
import itertools
import logging

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.config.experiment_config import ExperimentConfig
from src.distributed import DistributedParameters

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
    input_output_keys: list[str] = []
    ids: list[str] = []

    if distributed_parameters["distributed"]:
        model = model.module

    for batch in (
        tqdm(dataloader, desc="Generating")
        if distributed_parameters["rank"] == 0
        else dataloader
    ):
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in inspect.signature(model.forward).parameters
        }
        batch_generations = model.generate(
            **inputs,
            use_model_defaults=True,
            do_sample=False,
            num_beams=config.num_beams,
            max_length=1024,
        )
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
        input_output_keys.extend(batch["input-output"])
        ids.extend(batch["id"])

    # Gather all examples
    if distributed_parameters["distributed"]:
        all_generations = [None for _ in range(distributed_parameters["world_size"])]
        all_labels = [None for _ in range(distributed_parameters["world_size"])]
        all_input_output_keys = [
            None for _ in range(distributed_parameters["world_size"])
        ]
        all_ids = [None for _ in range(distributed_parameters["world_size"])]

        logger.info(
            f"[RANK {distributed_parameters['rank']}] Finished generation, entering gather"
        )
        torch.distributed.all_gather_object(all_generations, generations)
        torch.distributed.all_gather_object(all_labels, labels)
        torch.distributed.all_gather_object(all_input_output_keys, input_output_keys)
        torch.distributed.all_gather_object(all_ids, ids)

        # all_gather creates a list of lists, so we need to flatten
        all_generations = list(itertools.chain.from_iterable(all_generations))  # type:ignore
        all_labels = list(itertools.chain.from_iterable(all_labels))  # type:ignore
        all_input_output_keys = list(
            itertools.chain.from_iterable(all_input_output_keys)  # type:ignore
        )
        all_ids = list(itertools.chain.from_iterable(all_ids))  # type:ignore
    else:
        all_generations = generations
        all_labels = labels
        all_input_output_keys = input_output_keys
        all_ids = ids
    assert all(gen is not None for gen in all_generations)
    df = pd.DataFrame(
        [
            {
                "predicted": gen,
                "reference": label,
                "input_key": input_output_key.split("-")[0],
                "output_key": input_output_key.split("-")[1],
                "id": id,
            }
            for gen, label, input_output_key, id in zip(
                all_generations, all_labels, all_input_output_keys, all_ids
            )
        ]
    )
    # De-dupe, since with DDP we might have dupe generations
    df = df.drop_duplicates(
        subset=["id", "input_key", "output_key"], keep="first"
    ).reset_index(drop=True)
    return df
