import transformers
from torch.utils.data import DataLoader

from src.training.experiment_config import ExperimentConfig


def train(
    model,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
):
    optimizer = transformers.optimization.Adafactor(
        model.parameters(),
        warmup_init=True,
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer)

    for epoch in range(config.max_epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            out = model(**batch)
            breakpoint()
