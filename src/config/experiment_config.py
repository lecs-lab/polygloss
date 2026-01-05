import os
from dataclasses import dataclass, field
from typing import Literal

TRAIN_MODE = Literal["pretrain", "predict", "finetune", "lora", "grpo"]
MODEL_TYPE = Literal["seq2seq", "decoder"]
TASK_FORMAT = Literal[
    "multitask", "concatenated", "interleaved", "gloss-only", "segment-only"
]  # Format for glossing/segmentation task

_glotto_to_iso = {
    "arap1274": "arp",
    "gitx1241": "git",
    "dido1241": "ddo",
    "uspa1245": "usp",
    "nyan1302": "nyb",
    "natu1246": "ntu",
    "lezg1247": "lez",
}


@dataclass
class ExperimentConfig:
    # ============================
    # General
    # ============================

    mode: TRAIN_MODE
    """Training mode: 'pretrain' for pretraining on all data, 'finetune' for language-specific finetuning,
    or 'predict' for generating predictions on test data"""

    pretrained_model: str = "google/byt5-base"
    """Hugging Face model identifier for the pretrained model to use"""

    model_type: MODEL_TYPE = "seq2seq"
    """Architecture type: 'seq2seq' for encoder-decoder models or 'decoder' for decoder-only models"""

    new_hub_identifier: str | None = None
    """If provided, pushes the model to the HuggingFace hub"""

    # Dataset
    dataset_key: str = "lecslab/polygloss-corpus"
    """Hugging Face dataset identifier for the corpus to use"""

    glottocode: str | None = None
    """Glottocode of the language to finetune on (None for pretraining on all languages)"""

    task_format: TASK_FORMAT = "multitask"
    """Format for the joint seg/glossing"""

    max_tokens: int = 1024
    """Truncate prompts to this many tokens"""

    # ============================
    # Training
    # ============================

    max_epochs: int = 50
    """Maximum number of training epochs"""

    optimizer: str = "adafactor"
    """adamw | adafactor"""

    learning_rate: float = 5e-5
    """Learning rate for the optimizer"""

    min_learning_rate: float = 1e-5
    """Minimum learning rate when using LR decay"""

    weight_decay: float = 0.01
    """Weight decay for the optimizer"""

    use_warmup: bool = True
    """If true, will use linear LR warmup for first 3% of steps"""

    lr_schedule: Literal["none", "cosine"] = "cosine"
    """cosine | none"""

    grad_norm: float = 20.0
    """Max gradient norm"""

    batch_size: int = 64  # per gpu
    """Batch size per GPU for training and evaluation"""

    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps to simulate larger batch sizes"""

    models_dir: str | None = None
    """Directory to store checkpoints and models in. If not provided, use the same folder as the config file."""

    resume_from_checkpoint_id: str | None = None
    """WandB ID (and checkpoint ID) for checkpoint to resume training from."""

    lora_rank: int = 8
    """Rank for LoRa"""

    lora_dropout: float = 0.2
    """Dropout for LoRa"""

    lora_alpha: int = 8
    """Alpha for LoRa"""

    target_modules: str | None = None
    """Target modules for LoRA"""

    adapter_dir: str | None = None
    """LoRA adapter directory. If specified, will add adapter layer to pretrained model (for inference)"""

    # ============================
    # Generation
    # ============================

    num_beams: int = 2
    """Num beams for beam search"""

    grpo_group_size: int = 4
    """Num of generations for a GRPO group"""

    grpo_beta: float = 0.1

    grpo_temperature: float = 0.6

    grpo_top_p: float = 0.7

    # ============================
    # Computed properties
    # ============================
    @property
    def ft_isocode(self):
        if self.glottocode is not None:
            return _glotto_to_iso[self.glottocode]
        else:
            return None

    slurm_job_id: str | None = field(
        default_factory=lambda: os.environ.get("SLURM_JOB_ID"), init=False
    )

    def __post_init__(self):
        """Validates sanity checks on the parameters"""
        if self.glottocode is not None:
            if self.mode == "pretrain":
                raise ValueError("Pretraining should not have a specified glottocode!")
        else:
            if self.mode == "finetune":
                raise ValueError("Finetuning must have a glottocode!")
        if self.mode == "grpo" and self.task_format != "concatenated":
            raise ValueError("Can only do GRPO with `task_format=concatenated`")
