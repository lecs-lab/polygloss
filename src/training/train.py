import pathlib

import torch
import tqdm
from torch.utils.data import DataLoader

import wandb
from src.training.experiment_config import ExperimentConfig
from distributed import DistributedParameters


def train(
    model,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
    experiment_folder: pathlib.Path,
    distributed_parameters: DistributedParameters,
):
    # TODO:
    # [ ] multi gpu
    # [ ] early stopping
    # [x] mixed precision training
    # [ ] load best at end
    device = distributed_parameters["device"]
    pbar = tqdm.tqdm(total=config.max_epochs * len(train_dataloader), desc="Training")

    # TODO: Do we want to use adafactor over Adam?
    optimizer = torch.optim.Adafactor(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    # Load from checkpoint, if it exists
    start_epoch = 0
    if (experiment_folder / "checkpoint.pt").exists():
        print(
            "Loading from checkpoint. If you wanted to restart training from scratch, please delete the `checkpoint` directory."
        )
        checkpoint = torch.load(experiment_folder / "checkpoint.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    model.gradient_checkpointing_enable()
    scaler = torch.amp.grad_scaler.GradScaler()

    print(f"Training with {len(train_dataloader)} batches of size {config.batch_size}.")

    for epoch in range(start_epoch, config.max_epochs):
        if isinstance(train_dataloader.sampler, torch.utils.data.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)  # type:ignore

        # Train step
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast_mode.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = _get_loss(out, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach().item() / len(train_dataloader)
            pbar.update()

        # Eval step
        with (
            torch.amp.autocast_mode.autocast("cuda", dtype=torch.bfloat16),
            torch.inference_mode(),
        ):
            model.eval()
            eval_loss = 0.0
            for batch in tqdm.tqdm(dev_dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = _get_loss(out, batch["labels"])
                eval_loss += loss.detach().item() / len(dev_dataloader)

        # Log results
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
            experiment_folder / "checkpoint.pt",
        )

    # Save final model and remove checkpoint
    (experiment_folder / "checkpoint.pt").unlink()
    torch.save(model.state_dict(), experiment_folder / "model.pt")
    print(f"Saved model to {experiment_folder / 'model.pt'}")


def _get_loss(out, labels):
    """Dynamically gets the loss from the model outputs, which are different depending on the model"""

    if hasattr(out, "loss"):
        return out.loss
    elif hasattr(out, "logits"):
        raise NotImplementedError()
    else:
        raise NotImplementedError()
