import os
from socket import gethostname
from typing import TypedDict

import torch


class DistributedParameters(TypedDict):
    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    distributed: bool


def setup_ddp() -> DistributedParameters:
    if "WORLD_SIZE" in os.environ and "SLURM_PROCID" in os.environ:
        # Distributed
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        print(
            f"Hello from rank {rank} of {world_size} on {gethostname()} where there are"
            f" {gpus_per_node} allocated GPUs per node.",
            flush=True,
        )
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        if rank == 0:
            print(
                f"Group initialized? {torch.distributed.is_initialized()}", flush=True
            )
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return {
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "device": device,
            "distributed": True,
        }
    else:
        # Single GPU setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return {
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device": device,
            "distributed": False,
        }
