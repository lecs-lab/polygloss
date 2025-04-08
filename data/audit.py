"""Audits a dataset with various statistics"""

import typing

import datasets
from tqdm import tqdm


def audit(dataset: datasets.Dataset):
    stats = {
        "Missing segmentation": 0,
        "Missing translation": 0,
        "Missing glottocode": 0,
        "Missing metalang glottocode": 0,
        "Morpheme count mismatch": 0,
    }

    for row in tqdm(dataset, desc="Auditing"):
        row = typing.cast(typing.Mapping, row)
        if not row["segmentation"]:
            stats["Missing segmentation"] += 1
        if not row["translation"]:
            stats["Missing translation"] += 1
        if not row["glottocode"]:
            stats["Missing glottocode"] += 1
        if not row["metalang_glottocode"]:
            stats["Missing metalang glottocode"] += 1

    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
