import itertools
import logging
import pathlib

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.distributed import DistributedParameters
from src.training.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


def generate(
    model,
    tokenizer,
    dataloader: DataLoader,
    config: ExperimentConfig,
    experiment_folder: pathlib.Path,
    distributed_parameters: DistributedParameters,
) -> tuple[list[str], list[str] | None]:
    """Runs inference, generating predictions for each item in the dataloader.

    Returns:
        (tuple[list[str], list[str] | None]): The generated outputs and decoded labels (or None if not provided)
    """
    model.eval()
    device = distributed_parameters["device"]
    generations: list[str] = []
    labels: list[str | None] = []
    for batch in (
        tqdm(dataloader, desc="Generating")
        if distributed_parameters["rank"] == 0
        else dataloader
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        if distributed_parameters["distributed"]:
            model = model.module
        batch_generations = model.generate(
            **batch,
            use_model_defaults=True,
            do_sample=False,
            num_beams=5,
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
        torch.distributed.all_gather_object(all_generations, generations)
        torch.distributed.all_gather_object(all_labels, labels)
        all_generations = list(itertools.chain.from_iterable(all_generations))  # type:ignore
        all_labels = list(itertools.chain.from_iterable(all_labels))  # type:ignore
    else:
        all_generations = generations
        all_labels = labels

    assert all(gen is not None for gen in all_generations)
    if any(label is None for label in all_labels):
        all_labels = None
    return all_generations, all_labels  # type:ignore
