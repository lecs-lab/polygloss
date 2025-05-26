import pathlib

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.distributed import DistributedParameters
from src.training.experiment_config import ExperimentConfig


def generate(
    model,
    tokenizer,
    dataloader: DataLoader,
    config: ExperimentConfig,
    experiment_folder: pathlib.Path,
    distributed_parameters: DistributedParameters,
):
    """Runs inference, generating predictions for each item in the dataloader."""
    model.eval()
    device = distributed_parameters["device"]
    all_generations: list[str] = []
    all_labels: list[str] = []
    print("Generating...")
    for batch in (
        tqdm(dataloader) if distributed_parameters["rank"] == 0 else dataloader
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_generations = model.generate(
            **batch,
            use_model_defaults=True,
            do_sample=True,
            top_k=0,
            min_p=0.15,
            max_length=1024,
        )
        all_generations.extend(
            tokenizer.batch_decode(batch_generations, skip_special_tokens=True)
        )
        if "labels" in batch:
            all_labels.extend(
                tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            )
        break
    return all_generations, all_labels if len(all_labels) > 0 else None
