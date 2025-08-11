import argparse
import json
import logging
import pathlib
import pprint
import random
from dataclasses import asdict

import torch
from transformers.models.auto.modeling_auto import AutoModelForPreTraining
from transformers.models.auto.tokenization_auto import AutoTokenizer

import wandb
from src.config.config_to_dataclass import config_to_dataclass
from src.dataset import prepare_s2s_dataset
from src.distributed import DistributedParameters, setup_ddp
from src.evaluate import evaluate
from src.generate import generate
from src.train import ExperimentConfig, train

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run(
    config: ExperimentConfig,
    experiment_folder: pathlib.Path,
    distributed_parameters: DistributedParameters,
):
    random.seed(0)

    # Initialize WandB experiment
    if distributed_parameters["rank"] == 0:
        wandb.init(
            project="polygloss",
            entity="wav2gloss",
            config=asdict(config),
        )

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
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, use_fast=False)
    model = AutoModelForPreTraining.from_pretrained(config.pretrained_model).to(
        distributed_parameters["device"]
    )
    model.gradient_checkpointing_enable()
    if distributed_parameters["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[distributed_parameters["local_rank"]]
        )
    if config.model_type == "seq2seq":
        dataloaders = prepare_s2s_dataset.create_dataloaders(
            tokenizer=tokenizer,
            config=config,
            distributed_parameters=distributed_parameters,
        )
    else:
        raise NotImplementedError()

    if config.mode in ["pretrain", "finetune"]:
        train(
            model,
            tokenizer=tokenizer,
            train_dataloader=dataloaders["train"],
            dev_dataloader=dataloaders["dev"],
            config=config,
            experiment_folder=experiment_folder,
            models_folder=models_folder,
            distributed_parameters=distributed_parameters,
        )
    predictions = generate(
        model,
        tokenizer=tokenizer,
        dataloader=dataloaders["test"],
        config=config,
        distributed_parameters=distributed_parameters,
    )
    if distributed_parameters["rank"] == 0:
        wandb.log(
            {
                "predictions": wandb.Table(
                    columns=["predicted", "reference", "output_key"],
                    data=[[p.generation, p.label, p.output_key] for p in predictions],
                )
            }
        )

        # Evaluation (if we have labels, ie not in inference mode)
        if all(p.label is not None for p in predictions):
            metrics = evaluate(predictions)
            wandb.log(data={"test": metrics})
            with open(
                experiment_folder / "metrics.json", "w", encoding="utf-8"
            ) as file:
                json.dump(metrics, file, ensure_ascii=False, indent=4)
            logger.info(
                "Metrics logged to WandB and saved to %s",
                experiment_folder / "metrics.json",
            )
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
    folder = pathlib.Path(args.config).parent
    distributed_parameters = setup_ddp()
    run(
        config=config,
        experiment_folder=folder,
        distributed_parameters=distributed_parameters,
    )
    if distributed_parameters["distributed"]:
        torch.distributed.destroy_process_group()
