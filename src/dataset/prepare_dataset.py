"""Exposes `prepare_dataset`.
Creates a dataset of examples designed for either a seq2seq or decoder-only causal LM (specifically Qwen 3).

- Examples will be tokenized and have:
    - separate input and label sequences (seq2seq)
    - combined prompt+completion sequences, with prompt masked in loss calculation (decoder-only causal LM)
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
from transformers import AutoConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from data.model import boundary_pattern
from src.config.experiment_config import MODEL_TYPE
from src.distributed import DistributedParameters
from src.train import ExperimentConfig
from src.util.collator import FlexibleSeq2SeqCollator, FlexibleCausalLMCollator

logger = logging.getLogger(__name__)

supported_decoder_models = ["qwen3"]

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
                        label = _get_interleaved_segments(
                            row["id"], fields["segmentation"], fields["glosses"]
                        )
                        examples.append(
                            _create_example(
                                prompt, label, "t2sg", row, config.model_type
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

    model_config = AutoConfig.from_pretrained(config.pretrained_model)
    if model_config.is_encoder_decoder:
        _make_tokenizer = _make_seq2seq_tokenizer
    elif model_config.model_type in supported_decoder_models:
        # for now, only Qwen 3 is supported (uses chat template)
        _make_tokenizer = _make_causal_tokenizer_with_chat_template
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

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
    model_type: str,
):
    """Creates a dataloader for a dataset"""
    if model_type == "seq2seq":
        collator = FlexibleSeq2SeqCollator(tokenizer, label_pad_token_id=-100)
    elif model_type == "decoder":
        collator = FlexibleCausalLMCollator(tokenizer, mlm=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
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


def _make_seq2seq_tokenizer(tokenizer: PreTrainedTokenizerBase, max_length: int):
    """Tokenizer function for seq2seq models"""
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


def _make_causal_tokenizer_with_chat_template(
    tokenizer: PreTrainedTokenizerBase, 
    max_length: int,
    use_thinking: bool = False
):
    """Tokenizer function for decoder-only causal LMs using chat templates, specifically Qwen 3.    
    Args:
        tokenizer: The tokenizer instance
        max_length: Maximum sequence length
        use_thinking: Whether to enable Qwen 3's thinking mode (default: False)
    """
    def _tokenize(batch):
        nonlocal tokenizer, max_length, use_thinking
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        full_texts = []
        assistant_start_positions = []  # Track where assistant response starts
        
        for prompt, label in zip(batch["input"], batch["label"]):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": label}
            ]
            
            # Apply chat template
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=use_thinking,
            )
            full_texts.append(full_text)
            
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,  # True to get the assistant start token
                enable_thinking=use_thinking
                )
            assistant_start_positions.append(prompt_only)
        
        # Tokenize the full conversations
        model_inputs = tokenizer(
            full_texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        
        # Create labels with prompt tokens masked
        labels = []
        for i, prompt_only in enumerate(assistant_start_positions):
            # Tokenize the prompt-only part to find where assistant starts
            prompt_tokens = tokenizer(
                prompt_only,
                truncation=False,
                padding=False,
            )["input_ids"]
            
            prompt_length = len(prompt_tokens)
            full_length = len(model_inputs["input_ids"][i])
            
            # Mask prompt tokens, train on assistant tokens only
            label_seq = [-100] * prompt_length + model_inputs["input_ids"][i][prompt_length:]
            
            # Ensure label sequence matches input sequence length
            labels.append(label_seq[:full_length])
        
        model_inputs["labels"] = labels
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


def _get_interleaved_segments(id: str, segmentation_str: str, gloss_string: str):
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


def _create_example(
    prompt: str,
    label: str,
    task_key: str,
    row: typing.Mapping,
    model_type: MODEL_TYPE,
):
    if model_type in ["seq2seq", "decoder"]:
        return {
            "input": prompt,
            "label": label,
            "task": task_key,
            "id": row["id"],
            "glottocode": row["glottocode"],
        }
    else:
        raise NotImplementedError()
