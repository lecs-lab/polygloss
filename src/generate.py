import inspect
import logging

import pandas as pd
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
) -> pd.DataFrame | None:
    """Runs inference, generating predictions for each item in the dataloader.

    Critically, the `original_dataset` should be the dataset *prior to tokenization*,
    which is used to get metadata about each predicted example.
    It must be the same length and order as the dataloader.

    Returns:
        (list[PredictedExample]): The generated outputs and decoded labels (or None if not provided)
    """
    model.eval()
    device = distributed_parameters["device"]

    if distributed_parameters["distributed"]:
        model = model.module

    if distributed_parameters["rank"] == 0:
        generations: list[str] = []
        labels: list[str | None] = []
        task_keys: list[str] = []
        ids: list[str] = []

        for batch in tqdm(dataloader, desc="Generating"):
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

        assert all(gen is not None for gen in generations)
        df = pd.DataFrame(
            [
                {
                    "predicted": gen,
                    "reference": label,
                    "task": task_key,
                    "id": id,
                }
                for gen, label, task_key, id in zip(generations, labels, task_keys, ids)
            ]
        )
        return df
