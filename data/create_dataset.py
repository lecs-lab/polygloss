"""Creates the entire PolyGloss dataset from scratch by aggregating sources, filtering, and processing. May take a while."""

from .scrape_data import scrape_odin, scrape_sigmorphon_st

all_data = [*scrape_odin(), *scrape_sigmorphon_st()]

breakpoint()
