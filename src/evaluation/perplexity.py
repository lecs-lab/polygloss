import inspect
import logging
import math
from collections import defaultdict

import pandas as pd
import regex as re
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data.model import boundary_pattern
from src.config.experiment_config import ExperimentConfig
from src.distributed import DistributedParameters

logger = logging.getLogger(__name__)


def eval_ppl_per_lang(
    model,
    tokenizer,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
) -> pd.DataFrame | None:
    """Calculate loss and perplexity per language on the eval set."""
    if config.model_type == "seq2seq":
        pass
    else:
        raise NotImplementedError()

    model.eval()
    device = distributed_parameters["device"]
    forward_params = inspect.signature(
        (model.module if distributed_parameters["distributed"] else model).forward
    ).parameters

    if distributed_parameters["rank"] == 0:
        logger.info("Computing per-language perplexity...")
        loss_sum_per_language = defaultdict(float)
        num_tokens_per_language = defaultdict(int)
        num_morphemes_per_language = defaultdict(int)
        num_words_per_language = defaultdict(int)
        with (
            torch.amp.autocast_mode.autocast(
                distributed_parameters["device_type"], dtype=torch.bfloat16
            ),
            torch.inference_mode(),
        ):
            for batch in tqdm(
                dev_dataloader,
                desc="Evaluating",
            ):
                inputs = {
                    k: v.to(device) for k, v in batch.items() if k in forward_params
                }
                out = model(**inputs)
                # Compute loss without reducing so we can split up by language
                # Should be shape (batch_size,seq_length)
                labels = batch["labels"]
                losses = cross_entropy(
                    out.logits.permute(0, 2, 1),
                    labels.to(device),
                    ignore_index=-100,
                    reduction="none",
                )
                labels[labels == -100] = 0
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                for seq_losses, glottocode, seq_labels, label_text in zip(
                    losses, batch["glottocode"], labels, decoded_labels
                ):
                    if glottocode is None:
                        glottocode = "<unknown>"
                    loss_sum_per_language[glottocode] += (
                        seq_losses.sum().detach().item()
                    )
                    num_tokens_per_language[glottocode] += (  # type:ignore
                        torch.sum(seq_labels != 0).detach().item()
                    )
                    num_morphemes_per_language[glottocode] += len(
                        re.split(boundary_pattern, label_text)
                    )
                    num_words_per_language[glottocode] += len(label_text.split())

        glottocodes = sorted(
            set(c or "<unknown>" for c in dev_dataloader.dataset["glottocode"])
        )

        # Compute final dataframe
        rows = []
        for glottocode in glottocodes:
            if num_tokens_per_language[glottocode] == 0:
                print(f"Offender: {glottocode}")
            mean_loss = (
                loss_sum_per_language[glottocode] / num_tokens_per_language[glottocode]
            )
            lang_row = pd.Series(
                {
                    "glottocode": glottocode,
                    "loss": mean_loss,
                    "ppl": math.exp(mean_loss),
                    "num_tokens": num_tokens_per_language[glottocode],
                    "num_morphemes": num_morphemes_per_language[glottocode],
                    "num_words": num_words_per_language[glottocode],
                }
            )
            rows.append(lang_row)

        return pd.DataFrame(rows)
