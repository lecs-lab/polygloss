"""Creates the entire PolyGloss dataset from scratch by aggregating sources, filtering, and processing. May take a while."""

import datasets
import pandas as pd
from tqdm import tqdm

from data.audit import audit
from data.process import guess_metalang, standardize
from data.scrape_data import (
    scrape_cldf,
    scrape_gurani,
    scrape_imtvault,
    scrape_odin,
    scrape_sigmorphon_st,
)

# 1. Collate data from various sources
all_data = [
    *scrape_odin(),
    *scrape_sigmorphon_st(),
    *scrape_cldf(),
    *scrape_imtvault(),
    *scrape_gurani(),
]
print(f"Collated raw data with {len(all_data)} total examples.")

# 2. Process and standardize data
all_data = [standardize(ex) for ex in tqdm(all_data, desc="Standardizing")]
all_data = [guess_metalang(ex) for ex in tqdm(all_data, desc="Predicting metalangs")]

# 3. Remove close dupes
df = pd.DataFrame([vars(ex) for ex in all_data])
old_len = len(df)
df = df.drop_duplicates(subset=["transcription", "glosses", "glottocode"])
print(f"Removed {old_len - len(df)} duplicates for {len(df)} unique rows.")

# 4. Audit
dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
audit(dataset)

# 5. Push updated dataset
dataset.push_to_hub("lecslab/polygloss-corpus")
