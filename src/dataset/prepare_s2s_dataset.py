import os
import typing
from typing import cast

import datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.distributed import DistributedParameters
from src.train import ExperimentConfig
import regex

InputKey = typing.Literal["transcription", "segmentation"]
OutputKey = typing.Literal["segmentation", "glosses"]


def _read_toolbox_file(filepath: str) -> list[dict]:
   
    def _fix_punct(s: str):
        s = regex.sub(r"(\w)\?", r"\1 ?", s)
        s = regex.sub(r"(\w)\.(\s|$)", r"\1 .\2", s)
        s = regex.sub(r"(\w)\!", r"\1 !", s)
        s = regex.sub(r"(\w)\,", r"\1 ,", s)
        return s

    all_data = []
    with open(filepath, "r", encoding="utf-8") as file:
        current_entry = {"transcription": None, "segmentation": None, "glosses": None, "translation": None}
        skipped_lines = []

        for line in file:
            line = line.strip("\n")
            if not line.strip():
                continue

            prefix = line[:2]
            value = line[3:].strip() if len(line) > 3 else ""

            if prefix == "\\t":
                current_entry["transcription"] = _fix_punct(value)
            elif prefix == "\\m":
                current_entry["segmentation"] = _fix_punct(value)
            elif prefix == "\\g":
                current_entry["glosses"] = _fix_punct(regex.sub("\t", " ", value)) if value else None
            elif prefix == "\\l":
                current_entry["translation"] = value
                # End of entry
                all_data.append(current_entry.copy())
                current_entry = {"transcription": None, "segmentation": None, "glosses": None, "translation": None}
            elif prefix == "\\p":
                continue
            else:
                skipped_lines.append(line)

        # Handle last dangling entry
        if any(current_entry.values()):
            all_data.append(current_entry)

    if skipped_lines:
        print(f"[WARN] Skipped {len(skipped_lines)} malformed lines in {filepath}")
    return all_data


def create_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
) -> tuple[dict[str, DataLoader], datasets.DatasetDict]:
    """Creates dataloaders for training/finetuning.
    If `toolbox_dir` is set, loads from Toolbox files named `{glottocode}-{split}.txt`.
    """
    # ---- Load dataset from Toolbox files ----
    if getattr(config, "toolbox_dir", None):
        print(f"Loading Toolbox dataset for {config.glottocode} from {config.toolbox_dir}")
        dataset = datasets.DatasetDict()
        for split in ["train", "dev", "test"]:
            filename = f"{config.glottocode}-{split}.txt"
            path = os.path.join(config.toolbox_dir, filename)
            examples = _read_toolbox_file(path)
    
            for ex in examples:
                ex["glottocode"] = config.glottocode
                ex["language"] = config.glottocode
                ex["label"] = None
                ex["metalanguage"] = "English"

            dataset[split] = datasets.Dataset.from_list(examples)
    else:
        dataset = datasets.load_dataset(config.dataset_key)
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = _filter(dataset, config.glottocode)

    # ---- Create examples for each split ----
    for split in dataset:
        data_split = dataset[split]
        is_test_split = split == "test"

        # Apply sampling if needed
        if split == "train":
            if getattr(config, "limit", None) and getattr(config, "start_i", None):
                data_split = data_split.select(range(config.start_i, config.start_i + config.limit))
            elif getattr(config, "limit", None) and not getattr(config, "start_i", None):
                data_split = data_split.shuffle(seed=43).select(range(config.limit))
            elif getattr(config, "start_i", None) and not getattr(config, "limit", None):
                data_split = data_split.select(range(config.start_i, len(data_split)))

        examples = []
        for row in tqdm(data_split, desc=f"Creating examples for {split}"):
            row = typing.cast(typing.Mapping, row)
            has_gloss = bool(row.get("glosses")) and len(str(row["glosses"]).strip()) > 0

            # --- Unsegmented transcription examples ---
            if config.unsegmented_transcription:
                if is_test_split or has_gloss:
                    examples.append(
                        _create_example(
                            row,
                            input_key="transcription",
                            output_key="glosses",
                            use_translation=config.use_translation,
                        )
                    )

            # --- Segmented transcription examples ---
            if (
                config.segmented_transcription
                and row.get("segmentation")
                and (
                    not getattr(config, "exclude_st_segmented", False)
                    or (not row.get("source") == "sigmorphon_st")
                )
            ):
                if is_test_split or has_gloss:
                    examples.append(
                        _create_example(
                            row,
                            input_key="segmentation",
                            output_key="glosses",
                            use_translation=config.use_translation,
                        )
                    )

            # --- Segmentation prediction examples ---
            if config.create_segmentation_examples and (row.get("segmentation") or is_test_split):
                examples.append(
                    _create_example(
                        row,
                        input_key="transcription",
                        output_key="segmentation",
                        use_translation=False,
                    )
                )

        if not examples:
            print(f"[WARN] No examples created for split '{split}'")
        dataset[split] = datasets.Dataset.from_list(examples)

    print(dataset)

    # ---- Tokenize ----
    inputs_dataset = dataset.map(
        _make_tokenizer(tokenizer, max_length=config.max_tokens),
        batched=True,
        remove_columns=["input", "label", "output_key", "glottocode"],
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=typing.cast(int, tokenizer.pad_token_id)
    )

    # ---- Create dataloaders ----
    dataloaders = {}
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    for split in ["train", "dev", "test"]:
        if distributed_parameters["distributed"]:
            sampler = DistributedSampler(
                inputs_dataset[split],
                shuffle=(split == "train"),
                num_replicas=distributed_parameters["world_size"],
                rank=distributed_parameters["rank"],
            )
        else:
            sampler = RandomSampler(inputs_dataset[split]) if split == "train" else SequentialSampler(inputs_dataset[split])

        dataloaders[split] = DataLoader(
            inputs_dataset[split],
            batch_size=config.batch_size,
            collate_fn=collator,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders, dataset


def _filter(dataset: datasets.DatasetDict, glottocode: str | None):
    """Filter down to the relevant examples."""
    dataset = dataset.filter(lambda x: x["transcription"] is not None and x["glosses"] is not None)
    new_dataset = datasets.DatasetDict()

    if glottocode is not None:
        print(f"Filtering to {glottocode=}")
        dataset = dataset.filter(lambda row: row["glottocode"] == glottocode)
        if dataset["test"].num_rows == 0:
            raise ValueError(f"Dataset does not contain glottocode {glottocode}!")
        new_dataset = dataset
    else:
        print("Re-splitting dataset for pretraining")
        pretraining_data = datasets.concatenate_datasets([dataset["pretrain"], dataset["dev"]])
        pretraining_data = pretraining_data.train_test_split(test_size=0.1, seed=0)
        new_dataset["train"] = pretraining_data["train"]
        new_dataset["dev"] = pretraining_data["test"]
        new_dataset["test"] = dataset["test"]
    return new_dataset


def _make_tokenizer(tokenizer: PreTrainedTokenizerBase, max_length: int):
    def _tokenize(batch):
        model_inputs = tokenizer(
            batch["input"],
            text_target=batch.get("label"),
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
    """Creates an input prompt for seq2seq glossing or segmentation tasks."""
    input_seq = " ".join((row.get(input_key) or "").split())
    if output_key:
        output_seq = " ".join((row.get(output_key) or "").split())
    else:
        output_seq = None
    lang = row.get("language") or "an unknown language"

    prompt = f"Predict the {output_key} for the following {input_key} in {lang}."
    prompt += f"\n\nTranscription in {lang}: {input_seq}"

    if output_key == "glosses":
        prompt += f"\nIs transcription segmented: {input_key == 'segmentation'}"

    if use_translation and row.get("translation"):
        translation = " ".join((row["translation"]).split())
        prompt += f"\nTranslation in {row.get('metalanguage', 'unknown')}: {translation}"

    prompt += f"\n\n{output_key.capitalize()}: "

    return {
        "input": prompt,
        "label": output_seq,
        "output_key": output_key,
        "glottocode": row.get("glottocode"),
    }
