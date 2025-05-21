import os
from socket import gethostname
from typing import TypedDict

import torch


class DistributedParameters(TypedDict):
    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    device_type: str
    distributed: bool


def setup_ddp() -> DistributedParameters:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed (torchrun)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        print(
            f"Hello from rank {rank} of {world_size} on {gethostname()} "
            f"(local_rank {local_rank}, device {device})",
            flush=True,
        )

        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

        return {
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "device": device,
            "device_type": "cuda",
            "distributed": True,
        }
    else:
        # Single GPU setup
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        return {
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device": device,
            "device_type": str(device),
            "distributed": False,
        }
