import inspect
import logging
import tempfile

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
        generated_strings = tokenizer.batch_decode(
            batch_generations, skip_special_tokens=True
        )
        if config.model_type == "decoder":
            generated_strings = [
                s.replace(config.assistant_response_token, "")
                for s in generated_strings
            ]
        generations.extend(generated_strings)

        if "labels" in batch:
            batch["labels"][batch["labels"] == -100] = tokenizer.pad_token_id
            new_labels = tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )
            if config.model_type == "decoder":
                assert config.assistant_response_token is not None
                for i, lab in enumerate(new_labels):
                    if config.assistant_response_token in lab:
                        new_labels[i] = lab[
                            lab.index(config.assistant_response_token)
                            + len(config.assistant_response_token) :
                        ]
            labels.extend(new_labels)
        else:
            labels.extend([None] * len(batch_generations))
        task_keys.extend(batch["task"])
        ids.extend(batch["id"])

    # Gather all examples
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
    if distributed_parameters["distributed"]:
        logger.info(
            f"[RANK {distributed_parameters['rank']}] Finished generation, saving tempfile to disk"
        )
        tmp_dir = tempfile.gettempdir()
        tmp_path = f"{tmp_dir}/gen_{distributed_parameters['rank']}.parquet"
        df.to_parquet(tmp_path)
        torch.distributed.barrier()

        df = pd.concat(
            [
                pd.read_parquet(f"{tmp_dir}/gen_{i}.parquet")
                for i in range(distributed_parameters["world_size"])
            ],
            ignore_index=True,
        )

    # De-dupe, since with DDP we might have dupe generations due to weird batch sizes
    df = df.drop_duplicates(subset=["id", "task"], keep="first").reset_index(drop=True)
    return df
