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
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.distributed import DistributedParameters
from src.training.experiment_config import ExperimentConfig


def create_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
) -> dict[str, DataLoader]:
    """Creates dataloaders for training/finetuning

    Args:
        tokenizer (transformers.AutoTokenizer): The pretrained tokenizer
        config (ExperimentConfig): The experiment configuration
    """
    dataset = datasets.load_dataset(config.dataset_key)
    dataset = cast(datasets.DatasetDict, dataset)
    dataset = _filter(dataset, config.ft_glottocode)

    for split in dataset:
        examples = []
        for row in dataset[split]:
            row = typing.cast(typing.Mapping, row)
            if config.unsegmented_transcription:
                examples.append(
                    _create_example(
                        row,
                        input_key="transcription",
                        output_key="glosses",
                        use_translation=config.use_translation,
                    )
                )
            if (
                config.segmented_transcription
                and row["segmentation"]
                and (
                    (not config.exclude_st_segmented)
                    or (not row["source"] == "sigmorphon_st")
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
            if config.create_segmentation_examples and row["segmentation"]:
                examples.append(
                    _create_example(
                        row,
                        input_key="transcription",
                        output_key="segmentation",
                        use_translation=False,
                    )
                )
        dataset[split] = datasets.Dataset.from_list(examples)

    # Create prompts and tokenize
    dataset = dataset.map(
        _make_tokenizer(tokenizer, max_length=1024),
        batched=True,
        remove_columns=["input", "label"],
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=typing.cast(int, tokenizer.pad_token_id)
    )
    dataloaders = {}
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        # Default to a reasonable number when not in SLURM environment
        num_workers = 4
    for split in ["train", "dev", "test"]:
        if distributed_parameters["distributed"]:
            sampler = DistributedSampler(
                dataset[split],  # type:ignore
                shuffle=True,
                num_replicas=distributed_parameters["world_size"],
                rank=distributed_parameters["rank"],
            )
        else:
            sampler = (
                RandomSampler(dataset[split])
                if split == "train"
                else SequentialSampler(dataset[split])
            )
        dataloaders[split] = (
            DataLoader(
                dataset[split],  # type:ignore
                batch_size=config.batch_size,
                collate_fn=collator,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            ),
        )
    return dataloaders


def _filter(dataset: datasets.DatasetDict, glottocode: str | None):
    dataset = dataset.filter(
        lambda x: x["transcription"] is not None and x["glosses"] is not None
    )

    # Select the appropriate splits (ID or OOD)
    if glottocode is not None:
        print(f"Filtering to {glottocode=}")
        dataset = dataset.filter(lambda row: row["glottocode"] == glottocode)
        if dataset["dev_ID"].num_rows != 0:
            dataset["train"] = dataset["pretrain"]
            dataset["dev"] = dataset["dev_ID"]
            dataset["test"] = dataset["test_ID"]
        elif dataset["dev_OOD"].num_rows != 0:
            dataset["train"] = dataset["train_OOD"]
            dataset["dev"] = dataset["dev_OOD"]
            dataset["test"] = dataset["test_OOD"]
        else:
            raise ValueError("Neither ID nor OOD splits had your glottocode!")
    else:
        print("Re-splitting dataset for pretraining")
        # We must be pretraining
        # Instead of language-specific eval sets, let's use all of the pretraining and ID eval data and make iid splits
        pretraining_data = datasets.concatenate_datasets(
            [dataset["pretrain"], dataset["dev_ID"]]
        )
        pretraining_data = pretraining_data.train_test_split(test_size=0.1, seed=0)
        dataset["train"] = pretraining_data["train"]
        dataset["dev"] = pretraining_data["test"]
        dataset["test"] = dataset["test_ID"]
    return dataset


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
    input_key: typing.Literal["transcription", "segmentation"] = "transcription",
    output_key: typing.Literal["segmentation", "glosses"] = "glosses",
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
    return {"input": prompt, "label": output_seq}
