import inspect
import logging
import re
import os
import pathlib
import typing

import torch
import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForSeq2Seq

from config.experiment_config import ExperimentConfig
from src.distributed import DistributedParameters
from src.evaluate import DEFAULT_MORPHEME_BOUNDARIES

logger = logging.getLogger(__name__)

def calculate_ppl(
    model,
    tokenizer,
    dev_dataset: Dataset,
    models_folder: pathlib.Path,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
):
    """Calculate perplexity (per token/byte, morpheme, and word) on the eval set."""    
    model.eval()
    device = distributed_parameters["device"]
    
    if distributed_parameters["distributed"]:
        model = model.module
    
    if distributed_parameters["rank"] == 0:
        pbar = tqdm.tqdm(
            total=len(glottocodes), desc="Calculating PPL per language"
        )
    else:
        pbar = None
        
    forward_params = inspect.signature(
        (model.module if distributed_parameters["distributed"] else model).forward
    ).parameters
    
    boundary_pattern = re.compile("|".join(DEFAULT_MORPHEME_BOUNDARIES))
    
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        num_workers = 0

    # make dataloader per language
    glottocodes = dev_dataset.unique("glottocode")
    lang_dataloaders = {}
    for glottocode in glottocodes:
        lang_dataset = dev_dataset.filter(lambda x: x["glottocode"] == glottocode)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=typing.cast(int, tokenizer.pad_token_id)
        )
        dataloader = DataLoader(
            lang_dataset,
            collate_fn=data_collator,
            batch_size=config.batch_size,
            sampler=None
            if not distributed_parameters["distributed"]
            else torch.utils.data.distributed.DistributedSampler(
                lang_dataset,
                num_replicas=distributed_parameters["world_size"],
                rank=distributed_parameters["rank"],
                shuffle=False,
            ),
            num_workers=num_workers,
        )
        lang_dataloaders[glottocode] = dataloader
    
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
                
                bs = batch["labels"].size(0)
                loss = _get_loss(out, batch["labels"]).item()
                local_eval_loss_sum += loss * bs
                
                # Count tokens, morphemes, and words
                labels = batch["labels"]
                local_total_tokens += (labels != -100).sum().item()
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                for decoded_text in decoded_labels:
                    local_total_morphemes += len(re.split(boundary_pattern, decoded_text))
                    local_total_words += len(decoded_text.split())
            
            if distributed_parameters["distributed"]:
                local_loss_tensor = torch.tensor(local_eval_loss_sum, device=device)
                local_tokens_tensor = torch.tensor(local_total_tokens, device=device)
                local_morphemes_tensor = torch.tensor(local_total_morphemes, device=device)
                local_words_tensor = torch.tensor(local_total_words, device=device)
                
                torch.distributed.all_reduce(local_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(local_tokens_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(local_morphemes_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(local_words_tensor, op=torch.distributed.ReduceOp.SUM)

                eval_loss_sum = local_loss_tensor.item()
                total_tokens = local_tokens_tensor.item()
                total_morphemes = local_morphemes_tensor.item()
                total_words = local_words_tensor.item()
            else:
                eval_loss_sum = local_eval_loss_sum
                total_tokens = local_total_tokens
                total_morphemes = local_total_morphemes
                total_words = local_total_words
            
            eval_loss_per_language[glottocode] = eval_loss_sum
            tokens_per_language[glottocode] = total_tokens
            morphemes_per_language[glottocode] = total_morphemes
            words_per_language[glottocode] = total_words
            
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()
    
    # convert loss to perplexity
    ppl_per_language = {}
    for glottocode in glottocodes:
        loss = eval_loss_per_language[glottocode]
        ppl_per_language[glottocode] = {
            "ppl_per_token": torch.exp(torch.tensor(loss)).item() / tokens_per_language[glottocode],
            "ppl_per_morpheme": torch.exp(torch.tensor(loss)).item() / morphemes_per_language[glottocode],
            "ppl_per_word": torch.exp(torch.tensor(loss)).item() / words_per_language[glottocode],
        }
    
    return ppl_per_language

def _get_loss(out, labels):
    """Dynamically gets the loss from the model outputs, which are different depending on the model"""

    if hasattr(out, "loss"):
        return out.loss
    elif hasattr(out, "logits"):
        raise NotImplementedError()
    else:
        raise NotImplementedError()