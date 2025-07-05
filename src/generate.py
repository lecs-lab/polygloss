import inspect
import itertools
import logging
from dataclasses import dataclass

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.config.experiment_config import ExperimentConfig
from src.dataset.prepare_s2s_dataset import OutputKey, output_key_strings
from src.distributed import DistributedParameters

logger = logging.getLogger(__name__)


@dataclass
class PredictedExample:
    generation: str
    label: str | None
    output_key: OutputKey


def generate(
    model,
    tokenizer,
    dataloader: DataLoader,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
) -> list[PredictedExample]:
    """Runs inference, generating predictions for each item in the dataloader.

    Returns:
        (list[PredictedExample]): The generated outputs and decoded labels (or None if not provided)
    """
    model.eval()
    device = distributed_parameters["device"]

    generations: list[str] = []
    labels: list[str | None] = []
    output_keys: list[OutputKey] = []

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
            num_beams=1,
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
        if "output_key" in batch:
            output_keys.extend(
                [output_key_strings[index] for index in batch["output_key"].tolist()]
            )

    # Gather all examples
    if distributed_parameters["distributed"]:
        all_generations = [None for _ in range(distributed_parameters["world_size"])]
        all_labels = [None for _ in range(distributed_parameters["world_size"])]
        all_output_keys = [None for _ in range(distributed_parameters["world_size"])]
        logger.info(
            f"[RANK {distributed_parameters['rank']}] Finished generation, entering gather"
        )
        torch.distributed.all_gather_object(all_generations, generations)
        torch.distributed.all_gather_object(all_labels, labels)
        torch.distributed.all_gather_object(all_output_keys, output_keys)
        all_generations = list(itertools.chain.from_iterable(all_generations))  # type:ignore
        all_labels = list(itertools.chain.from_iterable(all_labels))  # type:ignore
        all_output_keys = list(itertools.chain.from_iterable(all_output_keys))  # type:ignore
    else:
        all_generations = generations
        all_labels = labels
        all_output_keys = output_keys

    assert all(gen is not None for gen in all_generations)

    return [
        PredictedExample(gen, label, output_key)
        for gen, label, output_key in zip(all_generations, all_labels, all_output_keys)
    ]
