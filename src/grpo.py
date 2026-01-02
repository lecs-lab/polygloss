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
            scores = torch.tensor(compute_scores(generations)).view(
                config.batch_size, config.grpo_group_size
            )
            # Normalize
            means = scores.mean(dim=-1).unsqueeze(1)
            stds = (scores.std(dim=-1) + 1e-9).unsqueeze(1)
            scores = (scores - means) / stds

        # Compute kl divergence
        inputs_repeated = {
            k: v.repeat_interleave(config.grpo_group_size, dim=0)
            if v is not None
            else v
            for k, v in batch.items()
        }
        mask = inputs_repeated["attention_mask"]
        policy_logprobs = torch.log_softmax(
            model(**inputs_repeated, decoder_input_ids=generated_ids), dim=-1
        )
        old_policy_logprobs = policy_logprobs.detach()
        log_ratio = policy_logprobs - old_policy_logprobs
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(
            min=1.0
        )
        log_importance_weights = log_importance_weights.unsqueeze(-1)
        coef_1 = torch.exp(log_importance_weights)
        with model.disable_adapter(), torch.no_grad():
            ref_logprobs = torch.log_softmax(
                model(**inputs_repeated, decoder_input_ids=generated_ids), dim=-1
            )

        breakpoint()
        # loss.backward()

        # Only update weights every accumulation_steps batches
        # if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
        #     torch.nn.utils.clip_grad_norm_(
        #         model.parameters(), max_norm=config.grad_norm
        #     )

        #     # Update LR as needed
        #     if step < total_warmup_steps:
        #         # Linear warmup
        #         new_lr = config.learning_rate * step / total_warmup_steps
        #     else:
        #         # Cosine decay
        #         progress = (step - total_warmup_steps) / (
        #             max_steps - total_warmup_steps
        #         )
        #         cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        #         new_lr = (
        #             config.min_learning_rate
        #             + (config.learning_rate - config.min_learning_rate) * cosine_decay
        #         )

        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = new_lr

        #     if distributed_parameters["rank"] == 0:
        #         wandb.log({"train/lr": new_lr}, step=step)

        #     optimizer.step()
        #     step += 1

        # # Note: multiply by accumulation_steps to get the actual loss value
        # num_tokens = torch.sum(batch["labels"] != -100).detach().item()
        # train_loss_sum += loss.item() * num_tokens * config.gradient_accumulation_steps
        # train_n += num_tokens

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
