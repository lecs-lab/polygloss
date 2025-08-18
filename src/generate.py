import inspect
import itertools
import logging
from dataclasses import dataclass
from typing import Mapping, cast

import torch
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.config.experiment_config import ExperimentConfig
from src.distributed import DistributedParameters

logger = logging.getLogger(__name__)


@dataclass
class PredictedExample:
    generation: str
    label: str | None


PredictionWithInfo = tuple[PredictedExample, Mapping]
"""A PredictedExample paired with the original row from the dataset"""


def generate(
    model,
    tokenizer,
    dataloader: DataLoader,
    original_dataset: Dataset,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
) -> list[PredictionWithInfo]:
    """Runs inference, generating predictions for each item in the dataloader.

    Critically, the `original_dataset` should be the dataset *prior to tokenization*,
    which is used to get metadata about each predicted example.
    It must be the same length and order as the dataloader.

    Returns:
        (list[PredictedExample]): The generated outputs and decoded labels (or None if not provided)
    """
    assert len(dataloader.dataset) == len(original_dataset)  # type:ignore

    model.eval()
    device = distributed_parameters["device"]

    generations: list[str] = []
    labels: list[str | None] = []

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
            labels.extend(
                tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            )
        else:
            labels.extend([None] * len(batch_generations))

    # Gather all examples
    if distributed_parameters["distributed"]:
        all_generations = [None for _ in range(distributed_parameters["world_size"])]
        all_labels = [None for _ in range(distributed_parameters["world_size"])]

        logger.info(
            f"[RANK {distributed_parameters['rank']}] Finished generation, entering gather"
        )
        torch.distributed.all_gather_object(all_generations, generations)
        torch.distributed.all_gather_object(all_labels, labels)
        # all_gather creates a list of lists, so we need to flatten
        all_generations = list(itertools.chain.from_iterable(all_generations))  # type:ignore
        all_labels = list(itertools.chain.from_iterable(all_labels))  # type:ignore
    else:
        all_generations = generations
        all_labels = labels
    assert all(gen is not None for gen in all_generations)
    return [
        (PredictedExample(gen, label), cast(Mapping, info))
        for gen, label, info in zip(all_generations, all_labels, original_dataset)
    ]
