"""Converts the original PolyGloss dataset into a dataset where segmented+unsegmented examples are combined"""

from typing import cast

import datasets
import pandas as pd

dataset = datasets.load_dataset(
    "lecslab/glosslm-corpus-split", revision="d6323debae1ddb6724b281331151991ba3abae8d"
)
dataset = cast(datasets.DatasetDict, dataset)


def combine_rows(group):
    print(group)
    breakpoint()
    pass


# First identify rows that have same ID
for split in dataset:
    df = cast(pd.DataFrame, dataset[split].to_pandas())
    df["segmentation"] = None
    df["id"] = df["id"].str.removesuffix("_unseg")
    original_size = len(df)

    # Find different rows with dupe IDs and assign unique IDs
    deduped = 0
    grouped = df.groupby("id")
    for name, group in grouped:
        if len(group) > 2:
            # We must have some dupes, assign unique IDs based on glosses
            new_ids = group["id"].str.cat(
                group["glosses"].factorize()[0].astype(str), sep="-"
            )
            df.loc[group.index, "id"] = new_ids
            deduped += 1
    print(f"Fixed {deduped} instances of duplicate IDs")

    # Remove complete duplicates
    df = cast(pd.DataFrame, df.drop_duplicates())
    print(f"Dropped {original_size - len(df)} exact duplicates")

    # Merge segmented and unsegmented rows
    original_size = len(df)
    merged_rows = 0

    grouped = df.groupby("id")
    for name, group in grouped:
        if len(group) == 1:
            if group["is_segmented"].item() == "yes":
                breakpoint()
                raise Exception()
        elif len(group) == 2:
            # Merge rows
            segmentation = group[group["is_segmented"] == "yes"]["transcription"].item()
            df.loc[group.index, "segmentation"] = segmentation
            merged_rows += 1
        elif len(group) > 2:
            breakpoint()
            raise Exception("Found more than 2 duplicates for ID:", name)

    # Filter out the originally segmented rows
    df = df[df["is_segmented"] != "yes"]
    df.drop(columns=["is_segmented"])
    print(
        f"Dropped {merged_rows}, resulting df has {len(df)}={original_size - merged_rows} rows"
    )
    if not isinstance(df, pd.DataFrame):
        raise ValueError()
    dataset[split] = datasets.Dataset.from_pandas(df)

dataset["pretrain"] = dataset["train"]
del dataset["train"]

dataset.push_to_hub(
    "lecslab/polygloss-split", commit_message="Create combined segmentation dataset"
)
