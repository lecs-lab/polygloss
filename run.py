import argparse
import os
import random
from dataclasses import asdict

import torch
import wandb
from transformers.models.auto.modeling_auto import AutoModelForPreTraining
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.config_to_dataclass import config_to_dataclass
from src.training.experiment_config import ExperimentConfig
from src.training.prepare_s2s_dataset import create_dataloaders
from src.training.train import train

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


def _make_if_needed(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def run(config: ExperimentConfig):
    random.seed(0)

    # Initialize WandB experiment
    if config.mode == "train" or config.mode == "finetune":
        run_name = config.exp_name
        if config.mode == "finetune":
            run_name += f"-ft-{config.ft_glottocode}"
        wandb.init(
            project="glossLM",
            entity="wav2gloss",
            name=run_name,
            config=asdict(config),
        )

    # Prepare model, dataset, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, use_fast=False)
    model = AutoModelForPreTraining.from_pretrained(config.pretrained_model)
    dataloaders = create_dataloaders(tokenizer=tokenizer, config=config)

    # Training loop
    if config.mode in ["pretrain", "finetune"]:
        model = train(
            model,
            train_dataloader=dataloaders["train"],
            dev_dataloader=dataloaders["dev"],
            config=config,
        )

    # args = transformers.Seq2SeqTrainingArguments(
    #     output_dir=output_dir,
    #     evaluation_strategy="epoch",
    #     learning_rate=config.learning_rate,
    #     per_device_train_batch_size=config.batch_size,
    #     per_device_eval_batch_size=1,
    #     eval_accumulation_steps=10,
    #     gradient_accumulation_steps=64,
    #     weight_decay=0.01,
    #     save_strategy="epoch",
    #     save_total_limit=10 if config.use_early_stopping else 3,
    #     num_train_epochs=config.max_epochs,
    #     predict_with_generate=True,
    #     load_best_model_at_end=config.use_early_stopping,
    #     logging_steps=100,
    #     generation_max_length=1024,
    #     generation_num_beams=3,
    #     report_to="wandb",
    #     metric_for_best_model="chrf++",
    #     fp16=True,
    #     dataloader_num_workers=4,
    # )
    # trainer = transformers.Seq2SeqTrainer(
    #     model,
    #     args,
    #     optimizers=(optimizer, lr_scheduler),
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
    #     ),
    #     train_dataset=dataset["train"],  # type:ignore
    #     eval_dataset=dataset["eval"],  # type:ignore
    #     compute_metrics=compute_metrics(tokenizer),
    #     tokenizer=tokenizer,
    #     callbacks=[
    #         DelayedEarlyStoppingCallback(
    #             early_stopping_patience=config.early_stopping_patience
    #         )
    #     ]
    #     if config.use_early_stopping
    #     else [],
    # )

    # if config.mode == "pretrain" or config.mode == "finetune":
    #     print("Training...")
    #     if config.checkpoint_path is not None:
    #         print(f"Continuing training from {config.checkpoint_path}")
    #     trainer.train(config.checkpoint_path)

    #     if config.output_model_path is None:
    #         raise ValueError("Must have an output path when training")
    #     print(f"Saving model to {config.output_model_path}")
    #     trainer.save_model(_make_if_needed(config.output_model_path))
    #     print("Model saved!")

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
    run(config)
