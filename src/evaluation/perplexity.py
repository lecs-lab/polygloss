import inspect
import logging
import re

import pandas as pd
import torch
import tqdm
from datasets import Dataset

from src.config.experiment_config import ExperimentConfig
from src.distributed import DistributedParameters
from src.evaluation.evaluate import DEFAULT_MORPHEME_BOUNDARIES

logger = logging.getLogger(__name__)


def eval_ppl_per_lang(
    model,
    tokenizer,
    dev_dataset: Dataset,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
) -> pd.DataFrame:
    """Calculate loss and perplexity per language on the eval set."""
    if config.model_type == "seq2seq":
        from src.dataset.prepare_s2s_dataset import create_dataloader
    else:
        raise NotImplementedError()

    model.eval()
    device = distributed_parameters["device"]
    if distributed_parameters["distributed"]:
        model = model.module

    forward_params = inspect.signature(model.forward).parameters
    boundary_pattern = re.compile("|".join(DEFAULT_MORPHEME_BOUNDARIES))

    # make dataloader per language
    glottocodes = dev_dataset.unique("glottocode")
    lang_dataloaders = {}
    for glottocode in glottocodes:
        lang_dataset = dev_dataset.filter(lambda x: x["glottocode"] == glottocode)
        lang_dataloaders[glottocode] = create_dataloader(
            dataset=lang_dataset,
            shuffle=False,
            batch_size=config.batch_size,
            tokenizer=tokenizer,
            distributed_parameters=distributed_parameters,
        )

    if distributed_parameters["rank"] == 0:
        pbar = tqdm.tqdm(total=len(glottocodes), desc="Calculating PPL per language")
    else:
        pbar = None
    with (
        torch.amp.autocast_mode.autocast(
            distributed_parameters["device_type"], dtype=torch.bfloat16
        ),
        torch.inference_mode(),
    ):
        eval_loss_per_language = {}
        tokens_per_language = {}
        morphemes_per_language = {}
        words_per_language = {}

        for glottocode, dev_dataloader in lang_dataloaders.items():
            local_total_tokens = 0
            local_total_morphemes = 0
            local_total_words = 0
            local_eval_loss_sum = 0.0

            for batch in dev_dataloader:
                keys_to_pop = [k for k in batch.keys() if k not in forward_params]
                for key in keys_to_pop:
                    batch.pop(key)
                batch = batch.to(device)
                out = model(**batch)

                loss = _get_loss(out, batch["labels"]).item()
                bs = batch["labels"].size(0)
                local_total_tokens += bs
                local_eval_loss_sum += loss * bs

                # Count tokens, morphemes, and words
                labels = batch["labels"]
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                for decoded_text in decoded_labels:
                    local_total_morphemes += len(
                        re.split(boundary_pattern, decoded_text)
                    )
                    local_total_words += len(decoded_text.split())

            if distributed_parameters["distributed"]:
                loss_and_counts = torch.tensor(
                    [
                        local_eval_loss_sum,
                        local_total_tokens,
                        local_total_morphemes,
                        local_total_words,
                    ],
                    device=device,
                )

                torch.distributed.all_reduce(
                    loss_and_counts, op=torch.distributed.ReduceOp.SUM
                )

                eval_loss_sum = loss_and_counts[0].item()
                total_tokens = loss_and_counts[1].item()
                total_morphemes = loss_and_counts[2].item()
                total_words = loss_and_counts[3].item()
            else:
                eval_loss_sum = local_eval_loss_sum
                total_tokens = local_total_tokens
                total_morphemes = local_total_morphemes
                total_words = local_total_words

            eval_loss_per_language[glottocode] = eval_loss_sum / total_tokens
            tokens_per_language[glottocode] = total_tokens
            morphemes_per_language[glottocode] = total_morphemes
            words_per_language[glottocode] = total_words

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    eval_rows = []
    for glottocode in glottocodes:
        loss = eval_loss_per_language[glottocode]
        lang_row = pd.Series(
            {
                "glottocode": glottocode,
                "loss": loss,
                "ppl": torch.exp(torch.tensor(loss)).item(),
                "num_tokens": tokens_per_language[glottocode],
                "num_morphemes": morphemes_per_language[glottocode],
                "num_words": words_per_language[glottocode],
            }
        )
        eval_rows.append(lang_row)
    eval_ppl_per_language = pd.DataFrame(eval_rows)

    return eval_ppl_per_language


def _get_loss(out, labels):
    """Dynamically gets the loss from the model outputs, which are different depending on the model"""

    if hasattr(out, "loss"):
        return out.loss
    elif hasattr(out, "logits"):
        raise NotImplementedError()
    else:
        raise NotImplementedError()
