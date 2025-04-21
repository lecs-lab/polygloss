import torch
import tqdm
import wandb
from torch.utils.data import DataLoader

from src.training.experiment_config import ExperimentConfig


def train(
    model,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
):
    pbar = tqdm.tqdm(total=config.max_epochs * len(train_dataloader), desc="Training")

    # TODO: Do we want to use adafactor over Adam?
    optimizer = torch.optim.Adafactor(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    for epoch in range(config.max_epochs):
        # Train step
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            out = model(**batch)
            loss = _get_loss(out, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.detach().item()
            pbar.update()

        # Eval step
        with torch.no_grad():
            model.eval()
            eval_loss = 0.0
            for batch in dev_dataloader:
                out = model(**batch)
                loss = _get_loss(out, batch["label_ids"])
                eval_loss += out.loss.detach().item()

        # Log results
        train_loss /= len(train_dataloader)
        eval_loss /= len(dev_dataloader)
        print(f"Epoch {epoch}\tLoss: {train_loss}\tEval loss: {eval_loss}")
        wandb.log(
            {"train/loss": train_loss, "train/epoch": epoch, "eval/loss": eval_loss},
            step=epoch * len(train_dataloader),
        )


def _get_loss(out, labels):
    """Dynamically gets the loss from the model outputs, which are different depending on the model"""

    if hasattr(out, "loss"):
        return out.loss
    elif hasattr(out, "logits"):
        raise NotImplementedError()
    else:
        raise NotImplementedError()
