"""Creates the entire PolyGloss dataset from scratch by aggregating sources, filtering, and processing. May take a while."""

import argparse
import logging
import typing
from collections import defaultdict

import datasets
import pandas as pd
from tqdm import tqdm

from data.audit import audit
from data.process import add_lang_info, standardize
from data.scrape_data import (
    scrape_cldf,
    scrape_fieldwork,
    scrape_guarani,
    scrape_imtvault,
    scrape_odin,
    scrape_sigmorphon_st,
)

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--message", "-m", help="Commit message")
args = parser.parse_args()

# 1. Collate data from various sources
logger.info("Collecting data")
all_data = [
    *scrape_odin(),
    *scrape_sigmorphon_st(),
    *scrape_cldf(),
    *scrape_imtvault(),
    *scrape_guarani(),
    *scrape_fieldwork(),
]
logger.info(f"Collected raw data with {len(all_data)} total examples.")

# 2. Process and standardize data
all_data = [standardize(ex) for ex in tqdm(all_data, desc="Standardizing")]
all_data = [
    add_lang_info(ex) for ex in tqdm(all_data, desc="Adding language information")
]

# 3. Remove close dupes
df = pd.DataFrame([vars(ex) for ex in all_data])
old_len = len(df)
df = df.drop_duplicates(subset=["transcription", "glosses", "glottocode"])
logger.info(f"Removed {old_len - len(df)} duplicates for {len(df)} unique rows.")

dataset = datasets.Dataset.from_pandas(df, preserve_index=False)

# 5. Split
dataset_dict = defaultdict(lambda: [])
for row in dataset:
    row = typing.cast(typing.Mapping, row)
    if row.get("designated_split") is not None:
        dataset_dict[row["designated_split"]].append(row)
    else:
        dataset_dict["train"].append(row)
dataset_dict = {
    key: datasets.Dataset.from_list(lst).remove_columns("designated_split")
    for key, lst in dataset_dict.items()
}
dataset_dict = datasets.DatasetDict(dataset_dict)

for split in dataset_dict:
    logger.info(f"Auditing '{split}'")
    audit(dataset_dict[split])

# 6. Push updated dataset
if args.message is not None:
    dataset_dict.push_to_hub("lecslab/polygloss-corpus", commit_message=args.message)
else:
    logger.warning("No message, not pushing to hub")
