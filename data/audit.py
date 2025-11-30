"""Audits a dataset with various statistics"""

import logging
import typing

import datasets
from tqdm import tqdm

from data.process import split_to_segments

logger = logging.getLogger(__name__)


def audit(dataset: datasets.Dataset):
    stats = {
        "Missing segmentation": 0,
        "Missing translation": 0,
        "Missing glottocode": 0,
        "Missing metalang glottocode": 0,
        "Word misalignment": 0,
        "Morpheme misalignment": 0,
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

        if row["segmentation"] and row["glosses"]:
            segments = split_to_segments(row["segmentation"])
            glosses = split_to_segments(row["glosses"])
            if len(segments) != len(glosses):
                # logger.warning(f"Word misalignment: {segments=} {glosses=}")
                stats["Word misalignment"] += 1
            elif any([len(s) != len(g) for s, g in zip(segments, glosses)]):
                # logger.warning(f"Morpheme misalignment: {segments=} {glosses=}")
                stats["Morpheme misalignment"] += 1

    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
