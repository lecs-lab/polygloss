# PolyGloss (formerly GlossLM)

A Massively Multilingual Corpus and Pretrained Model for Interlinear Glossed Text

[📄 Paper](https://arxiv.org/abs/2403.06399) | [📦 Models and data](https://huggingface.co/collections/lecslab/glosslm-66da150854209e910113dd87)

## Overview

PolyGloss consists of a multilingual corpus of IGT and pretrained models for interlinear glossing.

## Background

Interlinear Glossed Text (IGT) is a common format in language documentation projects. It looks something like this:

```
o sey x  -tok    r  -ixoqiil
o sea COM-buscar E3S-esposa
“O sea busca esposa.”
```

The three lines are as follows:

- The **transcription** line is a sentence in the target language (possibly segmented into morphemes)
- The **gloss** line gives an linguistic gloss for each morpheme in the transcription
- The **translation** line is a translation into a higher-resource language

Creating consistent and large IGT datasets is time-consuming and error-prone. The goal of **automated interlinear glossing** is to aid in this task by **predicting the gloss line given the transcription and translation line**.

## Using the dataset

```python
import datasets

glosslm_corpus = datasets.load_dataset("lecslab/glosslm-corpus")
```

## Using the model

```python
import transformers

# Your inputs
transcription = "o sey xtok rixoqiil"
translation = "O sea busca esposa."
lang = "Uspanteco"
metalang = "Spanish"
is_segmented = False

prompt = f"""Provide the glosses for the following transcription in {lang}.

Transcription in {lang}: {transcription}
Transcription segmented: {is_segmented}
Translation in {metalang}: {translation}\n
Glosses:
"""

model = transformers.T5ForConditionalGeneration.from_pretrained("lecslab/glosslm")
tokenizer = transformers.ByT5Tokenizer.from_pretrained("google/byt5-base", use_fast=False)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = tokenizer.batch_decode(model.generate(**inputs, max_length=1024), skip_special_tokens=True)
print(outputs[0]) # o sea COM-buscar E3S-esposa
```

## License

## Citation

```
@inproceedings{ginn-etal-2024-glosslm,
    title = "{G}loss{LM}: A Massively Multilingual Corpus and Pretrained Model for Interlinear Glossed Text",
    author = "Ginn, Michael  and
      Tjuatja, Lindia  and
      He, Taiqi  and
      Rice, Enora  and
      Neubig, Graham  and
      Palmer, Alexis  and
      Levin, Lori",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.683",
    pages = "12267--12286",
}
```

## Running Experiments

> [!NOTE]
> To reproduce the original GlossLM v1 experiments, please check out the `glosslm-v1` branch.

Interested in replicating or modifying our experiments? Pretraining, finetuning, and inference are handled with `run.py`, as follows:

```bash
# Python >=3.11
# Recommended to use venv for set up
python run.py some_config_file.cfg -o key1=val1 key2=val2
```

The run is defined by an INI-style _config file_. Each experiment should consist of a folder in `experiments` along with configuration files for pretraining, finetuning, or inference. See [an example experiment folder](experiments/base_igt). A config file looks something like:

```ini
[config]

mode = pretrain
pretrained_model = google/byt5-base

# Dataset
dataset_key = lecslab/polygloss-corpus

# Training
max_epochs = 50
learning_rate = 5e-5
batch_size = 16
```

The full list of possible options is in [experiment_config.py](src/training/experiment_config.py). In addition to the config file, you can specify any parameter overrides with `-o key1=val1 key2=val2`.

> [!NOTE]
> When running experiments over our evaluation languages, we use the paradigm shown in `run_curc.sh` where the config does not include a `glottocode` field and an override is used (e.g. `-o glottocode=uspa1245`).

## Development

### Set up environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements
```

### Testing
Tests should be written as functions in modules prefixed with `test_`, as in `test_evaluate_segmentation_example` in `evaluate.py`. Run tests with:

```bash
PYTHONPATH=. pytest
```

### Dataset
The full dataset creation is handled by `python create_dataset.py`. Any changes to the dataset should be incorporated in this script or its child scripts, so we have a definitive source of truth.
