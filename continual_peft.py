import argparse
import logging
import pathlib
import pprint
import torch

from src.config.config_to_dataclass import config_to_dataclass
from src.train import ExperimentConfig
from src.distributed import setup_ddp
from run import run 

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)

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

    if config.limit is None:
        raise ValueError("config.limit cannot be None for continual training")

    logger.info(f"Experiment config:\n{pprint.pformat(config)}")

    folder = pathlib.Path(args.config).parent

    total_size = 50  # ← replace with len(dataset)
    distributed_parameters = setup_ddp()
    config.limit = int(config.limit)
    for i in range(0, total_size, config.limit):
        if config.adapter_dir:
            logger.info(f"Starting chunk_{i}_{i + config.limit} with adapter path: {config.adapter_dir}")
        else:
            logger.info(f"Starting chunk {i}–{min(i + config.limit, total_size)}")

        config.start_i = i

        distributed_parameters = setup_ddp()

        out_folder = folder / f"{config.glottocode}" / f"chunk_{i}_{i + config.limit}"
        # try:
        out_folder.mkdir(exist_ok=True)
        # except FileExistsError:
        #     logger.info(f"Chunk {i}–{min(i + config.limit, total_size)} already exists. Skipping")
        #     print()
        out = run(config=config, experiment_folder=out_folder, distributed_parameters=distributed_parameters)


       
        if out and "final_model" in out:
            config.adapter_dir = out["final_model"]
        else:
            logger.warning("No models folder returned None — trying again from last chunk")

    if distributed_parameters.get("distributed"):
        torch.distributed.destroy_process_group()

    logger.info("All chunks processed successfully.")
