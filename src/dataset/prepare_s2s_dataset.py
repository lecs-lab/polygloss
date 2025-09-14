"""Exposes `prepare_dataset`, which creates a dataset of examples designed for a seq2seq model.

- Examples will be tokenized and have separate input and label sequences.
- Which kind of examples are created is determined by the options in `experiment_config.py`
"""

import os
import typing
from typing import cast

import datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.distributed import DistributedParameters
from src.train import ExperimentConfig
from src.util.collator import FlexibleSeq2SeqCollator

InputKey = typing.Literal["transcription", "segmentation"]
OutputKey = typing.Literal["segmentation", "glosses"]


def create_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
) -> tuple[dict[str, DataLoader], datasets.DatasetDict]:
    """Creates dataloaders for training/finetuning

    Args:
        tokenizer (transformers.AutoTokenizer): The pretrained tokenizer
        config (ExperimentConfig): The experiment configuration
    """
    dataset = datasets.load_dataset(config.dataset_key)
    dataset = cast(datasets.DatasetDict, dataset)
    dataset = _filter(dataset, config.glottocode)
    inputs_dataset = datasets.DatasetDict()

    for split in dataset:
        examples = []
        for row in tqdm(dataset[split], f"Creating examples for {split}"):
            row = typing.cast(typing.Mapping, row)
            if config.create_transcription_to_gloss == "train-test" or (
                split != "test" and config.create_transcription_to_gloss == "train-only"
            ):
                examples.append(
                    _create_example(
                        row,
                        input_key="transcription",
                        output_key="glosses",
                        use_translation=config.use_translation,
                    )
                )
            if row["segmentation"] and (
                config.create_segmentation_to_gloss == "train-test"
                or (
                    split != "test"
                    and config.create_segmentation_to_gloss == "train-only"
                )
            ):
                examples.append(
                    _create_example(
                        row,
                        input_key="segmentation",
                        output_key="glosses",
                        use_translation=config.use_translation,
                    )
                )
            if row["segmentation"] and (
                config.create_transcription_to_segmentation == "train-test"
                or (
                    split != "test"
                    and config.create_transcription_to_segmentation == "train-only"
                )
            ):
                examples.append(
                    _create_example(
                        row,
                        input_key="transcription",
                        output_key="segmentation",
                        use_translation=False,
                    )
                )
        inputs_dataset[split] = datasets.Dataset.from_list(examples)

    # Create prompts and tokenize
    inputs_dataset = inputs_dataset.map(
        _make_tokenizer(tokenizer, max_length=config.max_tokens),
        batched=True,
        remove_columns=["input", "label"],
        desc="Tokenizing",
    )

    collator = FlexibleSeq2SeqCollator(
        tokenizer, label_pad_token_id=typing.cast(int, tokenizer.pad_token_id)
    )
    dataloaders = {}
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        num_workers = 0
    for split in ["train", "dev", "test"]:
        if distributed_parameters["distributed"]:
            sampler = DistributedSampler(
                inputs_dataset[split],  # type:ignore
                shuffle=split == "train",
                num_replicas=distributed_parameters["world_size"],
                rank=distributed_parameters["rank"],
            )
        else:
            sampler = (
                RandomSampler(inputs_dataset[split])
                if split == "train"
                else SequentialSampler(inputs_dataset[split])
            )
        dataloaders[split] = DataLoader(
            inputs_dataset[split],  # type:ignore
            batch_size=config.batch_size,
            collate_fn=collator,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    return dataloaders, dataset


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


def _create_example(
    row: typing.Mapping,
    input_key: InputKey = "transcription",
    output_key: OutputKey = "glosses",
    use_translation: bool = True,
):
    """Creates an input prompt from the fields in the row.

    Works for either glossing (when `output_key` is `glosses`)
      or segmentation (when `output_key` is `segmentation`)
    """

    input_seq = " ".join((row[input_key]).split())
    output_seq = " ".join((row[output_key]).split())
    lang = (
        "an unknown language"
        if row["language"] == "" or not row["language"]
        else row["language"]
    )

    # Build the prompt
    prompt = f"Predict the {output_key} for the following {input_key} in {lang}."
    prompt += f"\n\nTranscription in {lang}: {input_seq}"

    if output_key == "glosses":
        prompt += f"\nIs transcription segmented: {input_key == 'segmentation'}"

    if use_translation and row["translation"] and len(row["translation"].strip()) > 0:
        translation = " ".join((row["translation"]).split())
        prompt += f"\nTranslation in {row['metalanguage'] or 'unknown'}: {translation}"

    prompt += f"\n\n{output_key.capitalize()}: "
    return {
        "input": prompt,
        "label": output_seq,
        "input-output": f"{input_key}-{output_key}",
        "id": row["id"],
        "glottocode": row["glottocode"],
    }
