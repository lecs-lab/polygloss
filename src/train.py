import inspect
import logging
import math
import pathlib

import torch
import tqdm
from peft import set_peft_model_state_dict
from torch.utils.data import DataLoader, DistributedSampler

import wandb
from src.config.experiment_config import ExperimentConfig
from src.distributed import DistributedParameters

logger = logging.getLogger(__name__)


def train(
    model,
    tokenizer,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
    experiment_folder: pathlib.Path,
    models_folder: pathlib.Path,
    distributed_parameters: DistributedParameters,
):
    """Training loop. Logs information to WandB and updates the model in place."""
    device = distributed_parameters["device"]

    if distributed_parameters["rank"] == 0:
        if not (run := wandb.run):
            raise Exception("WandB must be initialized!")
        run_id = run.id
    else:
        run_id = None

    if config.optimizer == "adafactor":
        optimizer = torch.optim.Adafactor(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )
    else:
        raise ValueError(f"Unrecognized optimizer: {config.optimizer}")

    start_epoch = 0
    step = 0
    max_steps = config.max_epochs * len(train_dataloader)
    if config.use_warmup:
        total_warmup_steps = int(0.03 * max_steps)
    else:
        total_warmup_steps = 0

    # Load from checkpoint, if it exists
    if config.resume_from_checkpoint_id:
        logger.info(f"Loading from checkpoint {config.resume_from_checkpoint_id}.")
        checkpoint = torch.load(
            models_folder / f"{config.resume_from_checkpoint_id}.checkpoint.pt",
            map_location=device,
        )
        model_to_load = model.module if distributed_parameters["distributed"] else model
        if config.mode == "lora":
            set_peft_model_state_dict(model_to_load, checkpoint["model_state_dict"])

        else:
            model_to_load.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        step = start_epoch * len(train_dataloader)

    if distributed_parameters["rank"] == 0:
        pbar = tqdm.tqdm(
            total=config.max_epochs * len(train_dataloader),
            desc="Training",
            initial=step,
        )
    else:
        pbar = None

    forward_params = inspect.signature(
        (model.module if distributed_parameters["distributed"] else model).forward
    ).parameters

    scaler = torch.amp.grad_scaler.GradScaler()
    logger.info(
        f"Training with {len(train_dataloader)} batches of size {config.batch_size}."
    )
    for epoch in range(start_epoch, config.max_epochs):
        if distributed_parameters["distributed"]:
            assert isinstance(train_dataloader.sampler, DistributedSampler)
            assert isinstance(dev_dataloader.sampler, DistributedSampler)
            train_dataloader.sampler.set_epoch(epoch)
            dev_dataloader.sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_dataloader:
            keys_to_pop = [k for k in batch.keys() if k not in forward_params]
            for key in keys_to_pop:
                batch.pop(key)
            batch = batch.to(device)
            optimizer.zero_grad()

            # Train in bfloat16
            with torch.amp.autocast_mode.autocast(
                distributed_parameters["device_type"], dtype=torch.bfloat16
            ):
                out = model(**batch)
                loss = _get_loss(out, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.grad_norm
            )

            # Update LR as needed
            if step < total_warmup_steps:
                # Linear warmup
                new_lr = config.learning_rate * step / total_warmup_steps
            else:
                # Cosine decay
                progress = (step - total_warmup_steps) / (
                    max_steps - total_warmup_steps
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                new_lr = (
                    config.min_learning_rate
                    + (config.learning_rate - config.min_learning_rate) * cosine_decay
                )
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            if distributed_parameters["rank"] == 0:
                wandb.log({"train/lr": new_lr}, step=step)

            scaler.step(optimizer)
            scaler.update()
            step += 1

            # Calculate loss, we will average over tokens
            num_tokens = torch.sum(batch["labels"] != -100).detach().item()
            train_loss_sum += loss.item() * num_tokens
            train_n += num_tokens

            if pbar:
                pbar.update()

        model.eval()
        logger.info("Evaluating...")
        with (
            torch.amp.autocast_mode.autocast(
                distributed_parameters["device_type"], dtype=torch.bfloat16
            ),
            torch.inference_mode(),
        ):
            eval_loss_sum = 0.0
            eval_n = 0
            for batch in dev_dataloader:
                keys_to_pop = [k for k in batch.keys() if k not in forward_params]
                for key in keys_to_pop:
                    batch.pop(key)
                batch = batch.to(device)
                out = model(**batch)
                loss = _get_loss(out, batch["labels"]).item()
                num_tokens = torch.sum(batch["labels"] != -100).detach().item()
                eval_loss_sum += loss * num_tokens
                eval_n += num_tokens

        # Sum losses over devices, if distributed
        if distributed_parameters["distributed"]:
            losses_tensor = torch.tensor(
                [train_loss_sum, train_n, eval_loss_sum, eval_n], device=device
            )
            torch.distributed.all_reduce(losses_tensor, torch.distributed.ReduceOp.SUM)
            train_loss = losses_tensor[0] / losses_tensor[1]
            eval_loss = losses_tensor[2] / losses_tensor[3]
        else:
            train_loss = train_loss_sum / train_n
            eval_loss = eval_loss_sum / eval_n

        if distributed_parameters["rank"] == 0:
            # Log results
            print(f"Epoch {epoch}\tLoss: {train_loss}\tEval loss: {eval_loss}")
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/epoch": epoch,
                    "eval/loss": eval_loss,
                },
                step=step,
            )
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": (
                        model.module if distributed_parameters["distributed"] else model
                    ).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                models_folder / f"{run_id}.checkpoint.pt",
            )

    # Save final model and remove checkpoint
    if distributed_parameters["rank"] == 0:
        (models_folder / f"{run_id}.checkpoint.pt").unlink(missing_ok=True)
        final_checkpoint_dir = models_folder / f"{run_id}.model"
        (
            model.module if distributed_parameters["distributed"] else model
        ).save_pretrained(final_checkpoint_dir)
        tokenizer.save_pretrained(final_checkpoint_dir)
        logger.info(f"Saved model to {final_checkpoint_dir.resolve()}")


def _get_loss(out, labels):
    """Dynamically gets the loss from the model outputs, which are different depending on the model"""

    if hasattr(out, "loss"):
        return out.loss
    elif hasattr(out, "logits"):
        raise NotImplementedError()
    else:
        raise NotImplementedError()
