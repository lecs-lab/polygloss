import argparse
import json
import logging
import pathlib
import pprint
import random
import sys
from dataclasses import asdict

import datasets
import torch
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data.dataloader import DataLoader
from transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

import wandb
from src.config.config_to_dataclass import config_to_dataclass
from src.dataset.prepare_dataset import create_dataloader, create_dataset
from src.distributed import DistributedParameters, setup_ddp
from src.evaluation.evaluate import evaluate
from src.evaluation.perplexity import eval_ppl_per_lang
from src.generate import generate
from src.train import ExperimentConfig, train
from src.util.pip_freeze import log_pip_freeze_artifact

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)


def run(
    config: ExperimentConfig,
    experiment_folder: pathlib.Path,
    distributed_parameters: DistributedParameters,
    dataset: datasets.DatasetDict,
):
    random.seed(0)

    # Initialize WandB experiment
    if distributed_parameters["rank"] == 0:
        if config.resume_from_checkpoint_id:
            wandb.init(
                project="polygloss",
                entity="lecs-general",
                config=asdict(config),
                id=config.resume_from_checkpoint_id,
                resume="must",
            )
        else:
            wandb.init(
                project="polygloss",
                entity="lecs-general",
                config=asdict(config),
            )

        # Log some other useful info
        api = HfApi()
        wandb.config.update(
            {
                "dataset.sha": api.dataset_info(config.dataset_key).sha,
                "python_version": sys.version,
            }
        )
        log_pip_freeze_artifact(f"pip-freeze-{wandb.run.id}")  # type:ignore

    if config.models_dir:
        models_folder = pathlib.Path(config.models_dir) / experiment_folder.stem
    else:
        models_folder = experiment_folder

    if config.glottocode is not None:
        # Create subfolders for each language if needed
        experiment_folder /= config.glottocode
        experiment_folder.mkdir(exist_ok=True)
        models_folder /= config.glottocode
    models_folder.mkdir(exist_ok=True, parents=True)

    # Prepare model, dataset, tokenizer
    if config.model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model).to(
            distributed_parameters["device"]
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config.pretrained_model).to(
            distributed_parameters["device"]
        )
    model.gradient_checkpointing_enable()
    if config.adapter_dir:
        model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, config.adapter_dir, is_trainable=True)
    elif config.mode == "lora":
        model.enable_input_require_grads()
        if config.model_type == "seq2seq":
            task_type = TaskType.SEQ_2_SEQ_LM
        elif config.model_type == "decoder":
            task_type = TaskType.CAUSAL_LM
        else:
            raise NotImplementedError()
        lora_config = LoraConfig(
            task_type=task_type,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        model = get_peft_model(model, lora_config)

    if distributed_parameters["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[distributed_parameters["local_rank"]]
        )

    dataloaders = {}
    dataloaders: dict[str, DataLoader] = {
        split: create_dataloader(
            dataset[split],
            split=split,
            config=config,
            tokenizer=tokenizer,
            distributed_parameters=distributed_parameters,
        )
        for split in dataset.keys()
    }

    if config.mode in ["pretrain", "finetune", "lora"]:
        key = "eval" if "eval" in dataloaders else "dev"
        train(
            model,
            tokenizer=tokenizer,
            train_dataloader=dataloaders["train"],
            dev_dataloader=dataloaders[key],
            config=config,
            models_folder=models_folder,
            distributed_parameters=distributed_parameters,
        )

    if distributed_parameters["distributed"]:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        model = model.module

    # Compute perplexity for each language
    key = "eval" if "eval" in dataloaders else "dev"
    perplexity_by_lang = eval_ppl_per_lang(
        model=model,
        tokenizer=tokenizer,
        dev_dataloader=dataloaders[key],
        config=config,
        distributed_parameters=distributed_parameters,
    )

    # Compute test predictions and metrics
    predictions = generate(
        model,
        tokenizer=tokenizer,
        dataloader=dataloaders["test"],
        config=config,
        distributed_parameters=distributed_parameters,
    )
    metrics = None
    if distributed_parameters["rank"] == 0:
        assert predictions is not None
        assert perplexity_by_lang is not None
        # Join with original dataset to add language info
        meta = (
            dataset["test"]  # type:ignore
            .to_pandas()[["id", "glottocode"]]
            .drop_duplicates(subset=["id"])
        )
        meta["id"] = meta["id"].astype(str)
        predictions["id"] = predictions["id"].astype(str)
        predictions_with_langs = predictions.merge(
            meta,
            on="id",
            how="left",
        )
        predictions.to_csv("wtf.csv")

        wandb.log({"predictions": wandb.Table(dataframe=predictions_with_langs)})

        lang_loss_table = wandb.Table(dataframe=perplexity_by_lang)
        wandb.log({"eval/loss_per_language": lang_loss_table})

        # Evaluation (if we have labels, ie not in inference mode)
        if predictions_with_langs["reference"].notnull().all():  # type:ignore
            metrics = evaluate(predictions_with_langs)
            wandb.log(data={"test": metrics})
            with open(
                experiment_folder / "metrics.json", "w", encoding="utf-8"
            ) as file:
                json.dump(metrics, file, ensure_ascii=False, indent=4)
            logger.info(
                "Metrics logged to WandB and saved to %s",
                experiment_folder / "metrics.json",
            )
    if distributed_parameters["distributed"]:
        if distributed_parameters["rank"] == 0:
            wandb.finish()
    else:
        wandb.finish()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="A config file (cfg, ini) with configuration parameters"
    )
    parser.add_argument(
        "--overrides",
        "-o",
        help="Override config arguments, in the format `key1=value1 key2=value2`",
        nargs="+",
    )
    args = parser.parse_args()
    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or [],
        dataclass_type=ExperimentConfig,
    )
    logger.info(f"Experiment config:\n{pprint.pformat(config)}")

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, use_fast=False)
    dataset = datasets.load_dataset(config.dataset_key)
    total_size = len(dataset["train"])  # type:ignore
    logger.info(f"TOTAL training samples = {total_size}")

    folder = pathlib.Path(args.config).parent
    chunk_sizes = [50, 100, 150, 200, 250, 300, 350, 380]

    for end_idx in chunk_sizes:
        config.train_subset_size = end_idx
        distributed_parameters = setup_ddp()
        dataset = create_dataset(tokenizer=tokenizer, config=config, end_idx=end_idx)
        logger.info(f"Starting chunk 0:{end_idx}")
        exp_folder = folder / f"chunk_0_{end_idx}"
        exp_folder.mkdir(exist_ok=True)
        if end_idx <= 100:
            config.max_epochs *= 2
        out = run(
            config=config,
            experiment_folder=exp_folder,
            distributed_parameters=distributed_parameters,
            dataset=dataset,
        )
        if distributed_parameters.get("distributed"):
            torch.distributed.destroy_process_group()

    logger.info("All chunks processed successfully.")
