"""Creates the entire PolyGloss dataset from scratch by aggregating sources, filtering, and processing. May take a while."""

from .scrape_data import (
    scrape_cldf,
    scrape_gurani,
    scrape_imtvault,
    scrape_odin,
    scrape_sigmorphon_st,
)

all_data = [
    *scrape_odin(),
    *scrape_sigmorphon_st(),
    *scrape_cldf(),
    *scrape_imtvault(),
    *scrape_gurani(),
]

print(f"Collated raw data with {len(all_data)} total examples.")

breakpoint()
