"""Exposes `prepare_dataset`, which creates a dataset of examples designed for a seq2seq model.

- Examples will be tokenized and have separate input and label sequences.
- Which kind of examples are created is determined by the options in `experiment_config.py`
"""

import logging
import os
import typing
from pathlib import Path
from string import Template
from typing import cast

import datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.config.experiment_config import MODEL_TYPE
from src.distributed import DistributedParameters
from src.train import ExperimentConfig
from src.util.collator import FlexibleSeq2SeqCollator

logger = logging.getLogger(__name__)

InputKey = typing.Literal["transcription", "segmentation"]
OutputKey = typing.Literal["segmentation", "glosses"]


def create_dataset(
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
) -> datasets.DatasetDict:
    """Creates dataset for training/finetuning

    Args:
        tokenizer (transformers.AutoTokenizer): The pretrained tokenizer
        config (ExperimentConfig): The experiment configuration
    """
    dataset = datasets.load_dataset(config.dataset_key)
    dataset = cast(datasets.DatasetDict, dataset)
    dataset = _filter(dataset, config.glottocode)
    inputs_dataset = datasets.DatasetDict()

    if "glosslm" in config.pretrained_model:
        logger.warning("Detected GlossLM base model, using GlossLM prompt.")

    # Make examples with different input/output combos
    for split in dataset:
        examples = []
        for row in tqdm(dataset[split], f"Creating examples for {split}"):
            row = typing.cast(typing.Mapping, row)
            fields = _prepare_prompt_fields(row)

            if config.task_format == "multitask":
                # Create transcription -> glosses and transcription -> segmentation for both splits
                # Create segmentation -> glosses for the train split ONLY
                if "glosslm" in config.pretrained_model:
                    prompt = _load_and_hydrate("glosslm.t2g", fields)
                else:
                    prompt = _load_and_hydrate("polygloss.multitask.t2g", fields)
                examples.append(
                    _create_example(
                        prompt, fields["glosses"], "t2g", row, config.model_type
                    )
                )
                if row["segmentation"]:
                    if "glosslm" not in config.pretrained_model:
                        prompt = _load_and_hydrate("polygloss.multitask.t2s", fields)
                        examples.append(
                            _create_example(
                                prompt,
                                fields["segmentation"],
                                "t2s",
                                row,
                                config.model_type,
                            )
                        )
                    if split != "test":
                        prompt = _load_and_hydrate("polygloss.multitask.s2g", fields)
                        examples.append(
                            _create_example(
                                prompt, fields["glosses"], "s2g", row, config.model_type
                            )
                        )
            elif config.task_format == "concatenated":
                # Create transcription -> glosses,
                #        segmentation -> glosses,
                #        transcription -> [segmentation, glosses] if possible
                if "glosslm" in config.pretrained_model:
                    raise NotImplementedError(
                        "GlossLM does not support the `concatenated` format"
                    )
                if fields["segmentation"]:
                    prompt = _load_and_hydrate("polygloss.concat.t2sg", fields)
                    label = f"{fields['segmentation']}\nGlosses: {fields['glosses']}"
                    examples.append(
                        _create_example(prompt, label, "t2sg", row, config.model_type)
                    )
                if split != "test":
                    prompt = _load_and_hydrate("polygloss.multitask.t2g", fields)
                    examples.append(
                        _create_example(
                            prompt, fields["glosses"], "t2g", row, config.model_type
                        )
                    )
                    if fields["segmentation"]:
                        prompt = _load_and_hydrate("polygloss.multitask.s2g", fields)
                        examples.append(
                            _create_example(
                                prompt, fields["glosses"], "s2g", row, config.model_type
                            )
                        )
            elif config.task_format == "interleaved":
                raise NotImplementedError()
            else:
                raise ValueError(f"Illegal value for task format: {config.task_format}")
        inputs_dataset[split] = datasets.Dataset.from_list(examples)

    # Create prompts and tokenize
    return inputs_dataset.map(
        _make_tokenizer(tokenizer, max_length=config.max_tokens),
        batched=True,
        remove_columns=["input", "label"],
        desc="Tokenizing",
    )


def create_dataloader(
    dataset: datasets.Dataset,
    shuffle: bool,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    distributed_parameters: DistributedParameters,
):
    """Creates a dataloader for a dataset"""
    collator = FlexibleSeq2SeqCollator(tokenizer, label_pad_token_id=-100)
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        num_workers = 0
    if distributed_parameters["distributed"]:
        sampler = DistributedSampler(
            dataset,  # type:ignore
            shuffle=shuffle,
            num_replicas=distributed_parameters["world_size"],
            rank=distributed_parameters["rank"],
        )
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset,  # type:ignore
        batch_size=batch_size,
        collate_fn=collator,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def _filter(dataset: datasets.DatasetDict, glottocode: str | None):
    """Filter down to the relevant examples (depending on pretraining vs finetuning)"""
    dataset = dataset.filter(
        lambda x: x["transcription"] is not None and x["glosses"] is not None
    )
    new_dataset = datasets.DatasetDict()

    # Select the appropriate splits (ID or OOD)
    if glottocode is not None:
        print(f"Filtering to {glottocode=}")
        dataset = dataset.filter(lambda row: row["glottocode"] == glottocode)
        if dataset["test"].num_rows == 0:
            raise ValueError(f"Dataset does not contain glottocode {glottocode}!")
        new_dataset = dataset
    else:
        print("Re-splitting dataset for pretraining")
        # We must be pretraining
        # Instead of language-specific eval sets, let's use all of the pretraining and ID eval data and make iid splits
        pretraining_data = datasets.concatenate_datasets(
            [dataset["train"], dataset["dev"]]
        )
        pretraining_data = pretraining_data.train_test_split(test_size=0.1, seed=0)
        new_dataset["train"] = pretraining_data["train"]
        new_dataset["dev"] = pretraining_data["test"]
        new_dataset["test"] = dataset["test"]
    return new_dataset


def _make_tokenizer(tokenizer: PreTrainedTokenizerBase, max_length: int):
    def _tokenize(batch):
        nonlocal tokenizer, max_length
        targets = batch.get("label")
        model_inputs = tokenizer(
            batch["input"],
            text_target=targets,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        return model_inputs

    return _tokenize


def _prepare_prompt_fields(row: typing.Mapping):
    """Given a row from the dataset, prepares the fields for the prompts"""
    transcription = " ".join((row["transcription"]).split())
    glosses = " ".join((row["glosses"]).split())
    if row["segmentation"] and len(row["segmentation"].strip()) > 0:
        segmentation = " ".join((row["segmentation"]).split())
    else:
        segmentation = None
    lang = (
        "an unknown language"
        if row["language"] == "" or not row["language"]
        else row["language"]
    )
    if (
        row["translation"]
        and len(row["translation"].strip()) > 0
        and row["translation"] != "Unknown"
    ):
        translation = " ".join((row["translation"]).split())
        metalang = row["metalanguage"] or "an unknown language"
    else:
        translation = "None"
        metalang = "English"
    return {
        "transcription": transcription,
        "glosses": glosses,
        "segmentation": segmentation,
        "translation": translation,
        "lang": lang,
        "metalang": metalang,
    }


def _load_and_hydrate(prompt_key: str, fields: dict):
    """Loads a prompt from a file and fills in fields"""
    prompt_path = Path(__file__).parent / (prompt_key + ".prompt")
    with open(prompt_path, "r") as prompt_file:
        prompt_text = prompt_file.read()
    prompt_template = Template(prompt_text)
    prompt = prompt_template.substitute(fields)
    return prompt


def _create_example(
    prompt: str,
    label: str,
    task_key: str,
    row: typing.Mapping,
    model_type: MODEL_TYPE,
):
    if model_type == "seq2seq":
        return {
            "input": prompt,
            "label": label,
            "task": task_key,
            "id": row["id"],
            "glottocode": row["glottocode"],
        }
    else:
        raise NotImplementedError()
