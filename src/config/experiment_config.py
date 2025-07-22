from dataclasses import dataclass
from typing import Literal

TRAIN_MODE = Literal["pretrain", "predict", "finetune"]
SEGMENTATION_MODE = Literal["segmented", "unsegmented", "both"]
MODEL_TYPE = Literal["seq2seq", "decoder"]

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

    # Dataset
    dataset_key: str = "lecslab/polygloss-corpus"
    """Hugging Face dataset identifier for the corpus to use"""

    glottocode: str | None = None
    """Glottocode of the language to finetune on (None for pretraining on all languages)"""

    segmented_transcription: bool = True
    """Whether to include examples with segmented transcriptions as input"""

    unsegmented_transcription: bool = True
    """Whether to include examples with unsegmented transcriptions as input"""

    exclude_st_segmented: bool = False
    """Whether to exclude segmented examples from the SIGMORPHON shared task"""

    create_segmentation_examples: bool = False
    """Whether to create examples for the segmentation task (transcription → segmentation)"""

    use_translation: bool = True
    """Whether to include translations in the input prompts when available"""

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

    weight_decay: float = 0.01
    """Weight decay for the optimizer"""

    batch_size: int = 64  # per gpu
    """Batch size per GPU for training and evaluation"""

    model_dir: str | None = None
    """Directory to store checkpoints and models in. If not provided, use the same folder as the config file."""

    # ============================
    # Generation
    # ============================

    num_beams: int = 2
    """Num beams for beam search"""

    # ============================
    # Computed properties
    # ============================
    @property
    def ft_isocode(self):
        if self.glottocode is not None:
            return _glotto_to_iso[self.glottocode]
        else:
            return None

    def __post_init__(self):
        """Validates sanity checks on the parameters"""
        if self.glottocode is not None:
            if self.mode == "pretrain":
                raise ValueError("Pretraining should not have a specified glottocode!")
        else:
            if self.mode != "pretrain":
                raise ValueError("Finetuning/prediction must have a glottocode!")
