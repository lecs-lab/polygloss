import argparse
import torch
from src.run import run  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="A config file (cfg, ini) with configuration parameters"
    )
    parser.add_argument(
        "--overrides",
        "-o",
        help="Override config arguments, in the format `key1=value1 key2=value2`",
        nargs="+",
    )
    args = parser.parse_args()
    config = config_to_dataclass(
        config_path=args.config,
        overrides=args.overrides or [],
        dataclass_type=ExperimentConfig,
    )
    logger.info(f"Experiment config:\n{pprint.pformat(config)}")
    total_size = len(config.dataset["train"])
    dataset = config.dataset["train"]
    folder = pathlib.Path(args.config).parent
    epochs = config.max_epochs
    chunk_size=100
        # Loop through dataset in chunks

i    )
    logger.info(f"Experiment config:\n{pprint.pformat(config)}")
    total_size = len(config.dataset["train"])
    dataset = config.dataset["train"]
    folder = pathlib.Path(args.config).parent
    epochs = config.max_epochs
    chunk_size=100
        # Loop through dataset in chunks

    #if i resume from checkpoint t restores the optimizer 
    #if i load put "adapter_dir" each time it is only the statedict
    wandb_run_id = None  # store W&B run ID across chunks
    i = 0
    for start_idx in range(0, total_size, 100):
        end_idx = min(start_idx + chunk_size, total_size)
        logger.info(f"Starting chunk {start_idx}–{end_idx}")
        if wandb_run_id is not None:
            config.resume_from_checkpoint_id = wandb_run_id
        config.max_epochs = epochs * (i + 1)
        # Setup distributed parameters
        distributed_parameters = setup_ddp()
        exp_folder= folder / f"chunk_{str(i)}"
        # Run training/evaluation for this chunk
        config.dataset["train"] = dataset[start_idx:end]
        out = run(
            config=config,
            experiment_folder=exp_folder,
            distributed_parameters=distributed_parameters,
        )

        if out is not None:
            wandb_run_id = out.get("wandb_run_id")
            logger.info(f"Resuming next chunk from W&B run ID: {wandb_run_id}")

        # Clean up DDP if needed
        if distributed_parameters.get("distributed"):
            torch.distributed.destroy_process_group()
        i+=1

    logger.info("All chunks processed successfully.")

