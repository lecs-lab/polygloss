import pathlib

import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

import wandb
from src.distributed import DistributedParameters
from src.training.experiment_config import ExperimentConfig


def train(
    model,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
    experiment_folder: pathlib.Path,
    distributed_parameters: DistributedParameters,
):
    # TODO:
    # [x] multi gpu
    # [ ] early stopping
    # [x] mixed precision training
    # [ ] load best at end
    device = distributed_parameters["device"]
    if distributed_parameters["rank"] == 0:
        pbar = tqdm.tqdm(
            total=config.max_epochs * len(train_dataloader), desc="Training"
        )
    else:
        pbar = None

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

    scaler = torch.amp.grad_scaler.GradScaler()
    print(f"Training with {len(train_dataloader)} batches of size {config.batch_size}.")
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
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast_mode.autocast(
                distributed_parameters["device_type"], dtype=torch.bfloat16
            ):
                out = model(**batch)
                loss = _get_loss(out, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Calculate loss. We want to sum up the losses PER example, even if batches are differently sized
            bs = batch["labels"].size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            if pbar:
                pbar.update()

        model.eval()
        print("Evaluating...")
        with (
            torch.amp.autocast_mode.autocast(
                distributed_parameters["device_type"], dtype=torch.bfloat16
            ),
            torch.inference_mode(),
        ):
            eval_loss_sum = 0.0
            eval_n = 0
            for batch in dev_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                bs = batch["labels"].size(0)
                eval_loss_sum += _get_loss(out, batch["labels"]).item() * bs
                eval_n += bs

        # Sum losses over devices, if distributed
        if distributed_parameters["distributed"]:
            losses_tensor = torch.tensor(
                [train_loss_sum, train_n, eval_loss_sum, eval_n], device=device
            )
            torch.distributed.all_reduce(losses_tensor, torch.distributed.ReduceOp.SUM)
            losses_tensor /= distributed_parameters["world_size"]

            if distributed_parameters["rank"] == 0:
                # Log results
                train_loss = losses_tensor[0] / losses_tensor[1]
                eval_loss = losses_tensor[2] / losses_tensor[3]
                print(f"Epoch {epoch}\tLoss: {train_loss}\tEval loss: {eval_loss}")
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/epoch": epoch,
                        "eval/loss": eval_loss,
                    },
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
    if distributed_parameters["rank"] == 0:
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
