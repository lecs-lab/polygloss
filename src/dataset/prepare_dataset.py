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
import regex as re
from glossing.igt import gloss_string_to_word_glosses
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from data.model import boundary_pattern
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
    templates = {
        key: _load(key)
        for key in [
            "glosslm.t2g",
            "polygloss.multitask.t2g",
            "polygloss.multitask.t2s",
            "polygloss.multitask.s2g",
            "polygloss.concat.t2sg",
            "polygloss.interleaved.t2sg",
        ]
    }
    for split in dataset:
        examples = []
        skipped = 0
        for row in tqdm(dataset[split], f"Creating examples for {split}"):
            row = typing.cast(typing.Mapping, row)
            fields = _prepare_prompt_fields(row)

            if config.task_format == "multitask":
                # Create transcription -> glosses and transcription -> segmentation for both splits
                # Create segmentation -> glosses for the train split ONLY
                if "glosslm" in config.pretrained_model:
                    prompt = templates["glosslm.t2g"].substitute(fields)
                else:
                    prompt = templates["polygloss.multitask.t2g"].substitute(fields)
                examples.append(
                    _create_example(
                        prompt, fields["glosses"], "t2g", row, config.model_type
                    )
                )
                if row["segmentation"]:
                    if "glosslm" not in config.pretrained_model:
                        prompt = templates["polygloss.multitask.t2s"].substitute(fields)
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
                        prompt = templates["polygloss.multitask.s2g"].substitute(fields)
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
                    prompt = templates["polygloss.concat.t2sg"].substitute(fields)
                    label = f"{fields['segmentation']}\nGlosses: {fields['glosses']}"
                    examples.append(
                        _create_example(prompt, label, "t2sg", row, config.model_type)
                    )
                if split != "test":
                    prompt = templates["polygloss.multitask.t2g"].substitute(fields)
                    examples.append(
                        _create_example(
                            prompt, fields["glosses"], "t2g", row, config.model_type
                        )
                    )
                    if fields["segmentation"]:
                        prompt = templates["polygloss.multitask.s2g"].substitute(fields)
                        examples.append(
                            _create_example(
                                prompt, fields["glosses"], "s2g", row, config.model_type
                            )
                        )
            elif config.task_format == "interleaved":
                # Create transcription -> glosses,
                #        segmentation -> glosses,
                #        transcription -> interleaved[segmentation, glosses] if possible
                if "glosslm" in config.pretrained_model:
                    raise NotImplementedError(
                        "GlossLM does not support the `interleaved` format"
                    )
                if fields["segmentation"]:
                    prompt = templates["polygloss.interleaved.t2sg"].substitute(fields)

                    # Make the interleaved label
                    try:
                        label = create_interleaved_segments(
                            row["id"], fields["segmentation"], fields["glosses"]
                        )
                        examples.append(
                            _create_example(
                                prompt,
                                label,
                                "t2sg_interleaved",
                                row,
                                config.model_type,
                            )
                        )
                    except ValueError as e:
                        if split == "test":
                            logger.warning(e)
                        skipped += 1

                if split != "test":
                    prompt = templates["polygloss.multitask.t2g"].substitute(fields)
                    examples.append(
                        _create_example(
                            prompt, fields["glosses"], "t2g", row, config.model_type
                        )
                    )
                    if fields["segmentation"]:
                        prompt = templates["polygloss.multitask.s2g"].substitute(fields)
                        examples.append(
                            _create_example(
                                prompt, fields["glosses"], "s2g", row, config.model_type
                            )
                        )
            else:
                raise ValueError(f"Illegal value for task format: {config.task_format}")
        logger.info(f"Skipped {skipped} for split {split}")
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


def _load(prompt_key: str):
    """Loads a prompt and returns a template"""
    prompt_path = Path(__file__).parent / (prompt_key + ".prompt")
    with open(prompt_path, "r") as prompt_file:
        prompt_text = prompt_file.read()
    return Template(prompt_text)


def create_interleaved_segments(id: str, segmentation_str: str, gloss_string: str):
    """Given a plain-text segmentation and gloss string, creates an interleaved string.

    For example, given:
        Segments: "the cat-s run"
        Glosses: "DET cat-PL run.SG"

    We would return
        "DET(the) cat(cat)-PL(s) run.SG(run)"
    """
    label = []
    segment_words = gloss_string_to_word_glosses(segmentation_str)
    gloss_words = gloss_string_to_word_glosses(gloss_string)

    if len(segment_words) != len(gloss_words):
        raise ValueError(
            f"Word mismatch! ID: {id}\nS: {segmentation_str}\nG: {gloss_string}"
        )

    for s_word, g_word in zip(segment_words, gloss_words):
        segments = re.split(boundary_pattern, s_word)
        glosses = re.split(boundary_pattern, g_word)
        if len(segments) != len(glosses):
            raise ValueError(
                f"Morpheme mismatch! ID: {id}\nS: {segmentation_str}\nG: {gloss_string}"
            )
        dividers = re.findall(boundary_pattern, g_word) + [""]
        word = "".join([f"{g}({s}){d}" for g, s, d in zip(glosses, segments, dividers)])
        label.append(word)
    label = " ".join(label)
    return label


def split_interleaved_segments(interleaved: str):
    """Given an interleaved string, splits into separate segment and gloss strings.

    For example, given:
        "DET(the) cat(cat)-PL(s) run.SG(run)"

    We would return
        Segments: "the cat-s run"
        Glosses: "DET cat-PL run.SG"
    """
    words = interleaved.split()
    segmentation_words = []
    gloss_words = []
    for word in words:
        interleaved_segments = re.split(boundary_pattern, word)
        dividers = re.findall(boundary_pattern, word) + [""]
        word_segments = []
        word_glosses = []
        for int_segment in interleaved_segments:
            match = re.match(r"(.*)\((.*)\)", int_segment)
            assert match
            word_glosses.append(match.group(1))
            word_segments.append(match.group(2))
        segmentation_words.append(
            "".join([f"{s}{d}" for s, d in zip(word_segments, dividers)])
        )
        gloss_words.append("".join([f"{g}{d}" for g, d in zip(word_glosses, dividers)]))
    segmentation = " ".join(segmentation_words)
    glosses = " ".join(gloss_words)
    return segmentation, glosses


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
