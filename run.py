import argparse
import pathlib
import random
from dataclasses import asdict

import torch
import wandb
from transformers.models.auto.modeling_auto import AutoModelForPreTraining
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.config_to_dataclass import config_to_dataclass
from src.distributed import DistributedParameters, setup_ddp
from src.training import prepare_s2s_dataset
from src.training.experiment_config import ExperimentConfig
from src.training.train import train


def run(
    config: ExperimentConfig,
    experiment_folder: pathlib.Path,
    distributed_parameters: DistributedParameters,
):
    random.seed(0)

    # Initialize WandB experiment
    if distributed_parameters["rank"] == 0 and (
        config.mode == "pretrain" or config.mode == "finetune"
    ):
        wandb.init(
            project="polygloss",
            entity="wav2gloss",
            config=asdict(config),
        )

    if config.ft_glottocode is not None:
        # Create subfolders for each language if needed
        experiment_folder /= config.ft_glottocode
        experiment_folder.mkdir(exist_ok=True)

    # Prepare model, dataset, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, use_fast=False)
    model = AutoModelForPreTraining.from_pretrained(config.pretrained_model).to(
        distributed_parameters["device"]
    )
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

    # Training loop
    if config.mode in ["pretrain", "finetune"]:
        model = train(
            model,
            train_dataloader=dataloaders["train"],
            dev_dataloader=dataloaders["dev"],
            config=config,
            experiment_folder=experiment_folder,
            distributed_parameters=distributed_parameters,
        )

    # elif config.mode == "predict":
    #     print("Creating predictions...")
    #     preds_dir = _make_if_needed(
    #         "../preds/{config.exp_name}/{force_unwrap(config.ft_isocode)}/"
    #     )
    #     preds_path = os.path.join(preds_dir, "test-preds.csv")
    #     preds = trainer.predict(dataset["test"])  # type:ignore
    #     labels = np.where(
    #         preds.predictions != -100, preds.predictions, tokenizer.pad_token_id
    #     )
    #     preds = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     preds = [strip_gloss_punctuation(pred) for pred in preds]
    #     gold = [strip_gloss_punctuation(g) for g in dataset["test"]["glosses"]]
    #     preds_df = pd.DataFrame(
    #         {
    #             "id": dataset["test"]["id"],
    #             "glottocode": dataset["test"]["glottocode"],
    #             "is_segmented": dataset["test"]["is_segmented"],
    #             "pred": preds,
    #             "gold": gold,
    #         }
    #     )
    #     preds_df.to_csv(preds_path, index=False)
    #     preds_df["pred"] = postprocess(preds_df["pred"])
    #     preds_df["gold"] = postprocess(preds_df["gold"])
    #     preds_df.to_csv(preds_path[:-4] + ".postprocessed.csv", index=False)
    #     print(f"Predictions for {config.ft_glottocode} data saved to {preds_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="A config file (cfg, ini) with configuration parameters"
    )
    parser.add_argument(
        "-o",
        "--overrides",
        help="Override config arguments, in the format `key1=value1 key2=value2`",
    )
    args = parser.parse_args()
    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or "",
        dataclass_type=ExperimentConfig,
    )
    folder = pathlib.Path(args.config).parent
    distributed_parameters = setup_ddp()
    run(
        config=config,
        experiment_folder=folder,
        distributed_parameters=distributed_parameters,
    )
    if distributed_parameters["distributed"]:
        torch.distributed.destroy_process_group()
