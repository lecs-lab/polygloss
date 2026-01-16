import inspect
import logging
import math
import pathlib

import torch
import tqdm
from peft import set_peft_model_state_dict
from torch.optim import Adafactor
from torch.optim.adamw import AdamW
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader, DistributedSampler

import wandb
from src.config.experiment_config import ExperimentConfig
from src.distributed import DistributedParameters
from src.grpo import grpo_epoch

logger = logging.getLogger(__name__)


def train(
    model,
    tokenizer,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
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
    max_steps = (
        config.max_epochs * len(train_dataloader) // config.gradient_accumulation_steps
    )
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

    logger.info(
        f"Training with {len(train_dataloader)} batches of size {config.batch_size}."
    )
    min_eval_loss = float("inf")
    since_best = 0
    for epoch in range(start_epoch, config.max_epochs):
        if distributed_parameters["distributed"]:
            assert isinstance(train_dataloader.sampler, DistributedSampler)
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        if config.mode in ["pretrain", "finetune", "lora"]:
            flag_tensor = torch.zeros(1).to(device)
            step, eval_loss = train_epoch(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                total_warmup_steps=total_warmup_steps,
                max_steps=max_steps,
                pbar=pbar,
                train_dataloader=train_dataloader,
                dev_dataloader=dev_dataloader,
                config=config,
                distributed_parameters=distributed_parameters,
            )
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                since_best = 0
                best_checkpoint_dir = models_folder / f"{run_id}.model"
                (
                    model.module if distributed_parameters["distributed"] else model
                ).save_pretrained(best_checkpoint_dir)
                tokenizer.save_pretrained(best_checkpoint_dir)
                logger.info(f"Saved model to {best_checkpoint_dir.resolve()}")
            else:
                since_best += 1
                if (
                    (epoch + 1) >= config.min_epochs
                    and config.early_stopping > 0
                    and since_best >= config.early_stopping
                ):
                    logger.info(
                        f"Early stopping. No improvements in the last {since_best} epochs"
                    )
                    flag_tensor += 1
            # If we early stopped, broadcast break to all ranks
            if distributed_parameters["distributed"]:
                torch.distributed.all_reduce(flag_tensor, op=ReduceOp.SUM)
            if flag_tensor.item() == 1:
                break
        elif config.mode == "grpo":
            step = grpo_epoch(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                total_warmup_steps=total_warmup_steps,
                max_steps=max_steps,
                pbar=pbar,
                train_dataloader=train_dataloader,
                dev_dataloader=dev_dataloader,
                config=config,
                distributed_parameters=distributed_parameters,
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
        if not final_checkpoint_dir.exists():
            # If the checkpoint already exists, we must have early stopped
            # If not, make it now
            (
                model.module if distributed_parameters["distributed"] else model
            ).save_pretrained(final_checkpoint_dir)
            tokenizer.save_pretrained(final_checkpoint_dir)
        logger.info(f"Saved model to {final_checkpoint_dir.resolve()}")

        if config.new_hub_identifier:
            (
                model.module if distributed_parameters["distributed"] else model
            ).push_to_hub(config.new_hub_identifier)
            tokenizer.push_to_hub(config.new_hub_identifier)
            logger.info(
                f"Pushed model and tokenizer to hub: {config.new_hub_identifier}"
            )


def train_epoch(
    model,
    tokenizer,
    optimizer: AdamW | Adafactor,
    epoch: int,
    step: int,
    total_warmup_steps: int,
    max_steps: int,
    pbar: tqdm.tqdm | None,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
):
    forward_params = inspect.signature(
        (model.module if distributed_parameters["distributed"] else model).forward
    ).parameters
    device = distributed_parameters["device"]
    train_loss_sum = 0.0
    train_n = 0

    for batch_idx, batch in enumerate(train_dataloader):
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
            loss = loss / config.gradient_accumulation_steps
        loss.backward()

        # Only update weights every accumulation_steps batches
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
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

            optimizer.step()
            step += 1

        # Note: multiply by accumulation_steps to get the actual loss value
        num_tokens = torch.sum(batch["labels"] != -100).detach().item()
        train_loss_sum += loss.item() * num_tokens * config.gradient_accumulation_steps
        train_n += num_tokens

        if pbar:
            pbar.update()

    if distributed_parameters["distributed"]:
        stats = torch.tensor(
            [train_loss_sum, train_n],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        train_loss_sum, train_n = stats.tolist()

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
    return step, eval_loss


def _get_loss(out, labels):
    """Dynamically gets the loss from the model outputs, which are different depending on the model"""

    if hasattr(out, "loss"):
        return out.loss
    elif hasattr(out, "logits"):
        raise NotImplementedError()
    else:
        raise NotImplementedError()
