import inspect
import logging
from pprint import pformat

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
    train_loss_sum = 0.0
    train_n = 0

    for batch_idx, batch in enumerate(train_dataloader):
        keys_to_pop = [k for k in batch.keys() if k not in forward_params]
        for key in keys_to_pop:
            batch.pop(key)
        batch = batch.to(device)
        optimizer.zero_grad()
        bs = batch["input_ids"].size(0)

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
                do_sample=True,
                temperature=0.6,
                top_p=0.7,
                repetition_penalty=1.05,
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
            ).view(bs, config.grpo_group_size)
            if batch_idx == 0:
                logger.info(
                    f"First train group: {pformat(list(zip(generations, compute_scores(generations)))[: config.grpo_group_size])}"
                )
            # Normalize
            means = scores.mean(dim=-1).unsqueeze(1)
            stds = (scores.std(dim=-1) + 1e-9).unsqueeze(1)
            scores = (scores - means) / stds
            scores = scores.view(bs * config.grpo_group_size).detach()

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
        token_counts = mask.sum(dim=-1).clamp(min=1)
        coef_1_seq = coef_1.sum(dim=-1) / token_counts
        coef_2_seq = coef_2.sum(dim=-1) / token_counts
        loss = -(coef_1_seq - coef_2_seq).mean()
        loss.backward()
        if step % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        train_loss_sum += loss.item()
        train_n += bs * config.grpo_group_size
        step += 1
        if pbar:
            pbar.update()

    logger.info("Evaluating...")
    with (
        torch.amp.autocast_mode.autocast(
            distributed_parameters["device_type"], dtype=torch.bfloat16
        ),
        torch.inference_mode(),
        torch.no_grad(),
    ):
        eval_reward_sum = 0.0
        eval_n = 0
        for batch_idx, batch in enumerate(dev_dataloader):
            keys_to_pop = [k for k in batch.keys() if k not in forward_params]
            for key in keys_to_pop:
                batch.pop(key)
            bs = batch["input_ids"].size(0)
            batch = batch.to(device)
            generated_ids = model.generate(
                **batch,
                do_sample=False,
                max_length=1024,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generations = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            scores = compute_scores(generations)
            if batch_idx == 0:
                logger.info(
                    "Some eval examples: " + pformat(list(zip(generations, scores))[:5])
                )
            eval_reward_sum += sum(scores)
            eval_n += bs

        if distributed_parameters["distributed"]:
            stats = torch.tensor(
                [train_loss_sum, train_n, eval_reward_sum, eval_n],
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            train_loss_sum, train_n, eval_reward_sum, eval_n = stats.tolist()

        train_loss = train_loss_sum / train_n
        eval_reward = eval_reward_sum / eval_n

        # Log results
        print(f"Epoch {epoch}\tLoss: {train_loss}\tEval reward: {eval_reward}")
        wandb.log(
            {
                "train/loss": train_loss,
                "train/epoch": epoch,
                "eval/avg_reward": eval_reward,
            },
            step=step,
        )
    return step


def compute_scores(generations: list[str]):
    gloss_label = "\nGlosses: "  # Split on this label
    scores = []
    for gen in generations:
        splits = gen.split(gloss_label)
        if len(splits) <= 1:
            logger.warning(f"No splits: {gen}")
            scores.append(0)
            continue
        if len(splits) > 2:
            logger.warning(f"Too many splits: {gen}")
            scores.append(0)
            continue
        segments, glosses = splits
        scores.append(alignment_score([(segments, glosses)]))
    return scores
