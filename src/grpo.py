import inspect
import logging

import torch
import tqdm
from torch.optim import Adafactor
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

import wandb
from src.config.experiment_config import ExperimentConfig
from src.distributed import DistributedParameters
from src.evaluation.alignment_score import alignment_score

logger = logging.getLogger(__name__)


def grpo_epoch(
    model,
    tokenizer,
    optimizer: AdamW | Adafactor,
    epoch: int,
    step: int,
    total_warmup_steps: int,
    max_steps: int,
    pbar: tqdm.tqdm | None,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    config: ExperimentConfig,
    distributed_parameters: DistributedParameters,
):
    forward_params = inspect.signature(
        (model.module if distributed_parameters["distributed"] else model).forward
    ).parameters
    device = distributed_parameters["device"]

    for batch_idx, batch in enumerate(train_dataloader):
        keys_to_pop = [k for k in batch.keys() if k not in forward_params]
        for key in keys_to_pop:
            batch.pop(key)
        batch = batch.to(device)
        optimizer.zero_grad()

        # Generate step
        with (
            torch.no_grad(),
            torch.amp.autocast_mode.autocast(
                distributed_parameters["device_type"], dtype=torch.bfloat16
            ),
        ):
            model.eval()
            generated_ids = model.generate(
                **batch,
                use_model_defaults=True,
                do_sample=True,
                num_return_sequences=config.grpo_group_size,
                max_length=1024,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generations = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            decoder_in = generated_ids[:, :-1]
            labels = generated_ids[:, 1:]
            scores = torch.tensor(
                compute_scores(generations), device=generated_ids.device
            ).view(config.batch_size, config.grpo_group_size)
            # Normalize
            means = scores.mean(dim=-1).unsqueeze(1)
            stds = (scores.std(dim=-1) + 1e-9).unsqueeze(1)
            scores = (scores - means) / stds
            scores = scores.view(config.batch_size * config.grpo_group_size).detach()

        # Compute kl divergence
        inputs_repeated = {
            k: v.repeat_interleave(config.grpo_group_size, dim=0)
            if v is not None
            else v
            for k, v in batch.items()
        }
        mask = labels != tokenizer.pad_token_id
        with torch.no_grad():
            old_policy_logprobs = (
                torch.log_softmax(
                    model(**inputs_repeated, decoder_input_ids=decoder_in).logits,
                    dim=-1,
                )
                .gather(dim=-1, index=labels.unsqueeze(-1))
                .squeeze(-1)
                .detach()
            )
        policy_logprobs = (
            torch.log_softmax(
                model(**inputs_repeated, decoder_input_ids=decoder_in).logits, dim=-1
            )
            .gather(dim=-1, index=labels.unsqueeze(-1))
            .squeeze(-1)
        )
        log_ratio = policy_logprobs - old_policy_logprobs
        coef_1 = torch.exp(log_ratio) * scores.unsqueeze(1) * mask
        # log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(
        #     min=1.0
        # )
        # log_importance_weights = log_importance_weights.unsqueeze(-1)

        with model.disable_adapter(), torch.no_grad():
            ref_logprobs = (
                torch.log_softmax(
                    model(**inputs_repeated, decoder_input_ids=decoder_in).logits,
                    dim=-1,
                )
                .gather(dim=-1, index=labels.unsqueeze(-1))
                .squeeze(-1)
                .detach()
            )
        log_ref_ratio = ref_logprobs - policy_logprobs
        coef_2 = (
            config.grpo_beta * (torch.exp(log_ref_ratio) - log_ref_ratio - 1) * mask
        )
        loss = (
            -1
            * torch.sum(coef_1 + coef_2)
            / (config.batch_size * config.grpo_group_size)
        )
        loss.backward()

        if pbar:
            pbar.update()

    if distributed_parameters["distributed"]:
        stats = torch.tensor(
            [train_loss_sum, train_n],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        train_loss_sum, train_n = stats.tolist()

    if distributed_parameters["rank"] == 0:
        model.eval()
        logger.info("Evaluating...")
        with (
            torch.amp.autocast_mode.autocast(
                distributed_parameters["device_type"], dtype=torch.bfloat16
            ),
            torch.inference_mode(),
        ):
            eval_loss_sum = 0.0
            eval_n = 0
            for batch in dev_dataloader:
                keys_to_pop = [k for k in batch.keys() if k not in forward_params]
                for key in keys_to_pop:
                    batch.pop(key)
                batch = batch.to(device)
                out = model(**batch)
                loss = _get_loss(out, batch["labels"]).item()
                num_tokens = torch.sum(batch["labels"] != -100).detach().item()
                eval_loss_sum += loss * num_tokens
                eval_n += num_tokens

            train_loss = train_loss_sum / train_n
            eval_loss = eval_loss_sum / eval_n
        # Log results
        print(f"Epoch {epoch}\tLoss: {train_loss}\tEval loss: {eval_loss}")
        wandb.log(
            {
                "train/loss": train_loss,
                "train/epoch": epoch,
                "eval/loss": eval_loss,
            },
            step=step,
        )
    return step


def compute_scores(generations: list[str]):
    gloss_label = "\nGlosses: "  # Split on this label
    scores = []
    for gen in generations:
        splits = gen.split(gloss_label)
        if len(splits) == 0:
            scores.append(0)
            continue
        if len(splits) > 2:
            breakpoint()
        segments, glosses = splits
        scores.append(alignment_score([(segments, glosses)]))
    return scores
