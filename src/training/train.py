import pathlib

import torch
import tqdm
from torch.utils.data import DataLoader

import wandb
from src.training.experiment_config import ExperimentConfig


def train(
    model,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
    experiment_folder: pathlib.Path,
    device: str,
):
    # TODO:
    # [ ] early stopping
    # [x] mixed precision training
    # [ ] load best at end

    pbar = tqdm.tqdm(total=config.max_epochs * len(train_dataloader), desc="Training")

    # TODO: Do we want to use adafactor over Adam?
    optimizer = torch.optim.Adafactor(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    # Load from checkpoint, if it exists
    start_epoch = 0
    if (experiment_folder / "checkpoint").exists():
        print(
            "Loading from checkpoint. If you wanted to restart training from scratch, please delete the `checkpoint` directory."
        )
        checkpoint = torch.load(experiment_folder / "checkpoint", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    model.gradient_checkpointing_enable()
    scaler = torch.amp.grad_scaler.GradScaler(device)

    print(f"Training with {len(train_dataloader)} batches of size {config.batch_size}.")

    for epoch in range(start_epoch, config.max_epochs):
        # Train step
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast_mode.autocast(device, dtype=torch.float16):
                out = model(**batch)
                loss = _get_loss(out, batch["labels"])
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach().item()
            pbar.update()

        # Eval step
        with (
            torch.amp.autocast_mode.autocast(device, dtype=torch.float16),
            torch.inference_mode(),
        ):
            model.eval()
            eval_loss = 0.0
            for batch in tqdm.tqdm(dev_dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = _get_loss(out, batch["labels"])
                eval_loss += loss.detach().item()

        # Log results
        train_loss /= len(train_dataloader)
        eval_loss /= len(dev_dataloader)
        print(f"Epoch {epoch}\tLoss: {train_loss}\tEval loss: {eval_loss}")
        wandb.log(
            {"train/loss": train_loss, "train/epoch": epoch, "eval/loss": eval_loss},
            step=epoch * len(train_dataloader),
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            experiment_folder / "checkpoint",
        )

    # Save final model and remove checkpoint
    (experiment_folder / "checkpoint").rmdir()
    torch.save(model.state_dict(), experiment_folder / "model")
    print(f"Saved model to {experiment_folder / 'model'}")


def _get_loss(out, labels):
    """Dynamically gets the loss from the model outputs, which are different depending on the model"""

    if hasattr(out, "loss"):
        return out.loss
    elif hasattr(out, "logits"):
        raise NotImplementedError()
    else:
        raise NotImplementedError()
