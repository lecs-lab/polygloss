import pandas as pd

import wandb
from data.lang_codes import glotto_to_iso3
from data.scrape_data import evaluation_languages

api = wandb.Api()

metrics = {
    "glossing.morphemes.error_rate": "Glossing MER",
    "segmentation.f1": "Segmentation F1",
    "alignment": "Alignment",
}
runs = [
    "polygloss-byt5-multitask (only long sentences)",
    "polygloss-byt5-concat (only long sentences)",
    "polygloss-byt5-interleaved (only long sentences)",
]
nice_names = [
    "PolyGloss [multitask, long]",
    "PolyGloss [concat, long]",
    "PolyGloss [interleaved, long]",
]

results = []

for run in api.runs(path="lecs-general/polygloss"):
    if run.name in runs:
        for metric in metrics:
            for lang in evaluation_languages:
                score = run.summary_metrics["test"][lang]
                for key in metric.split("."):
                    if key not in score:
                        score = None
                        break
                    score = score[key]
                score = round(score, 3)  # type:ignore
                results.append(
                    {
                        "Method": nice_names[runs.index(run.name)],
                        "Language": glotto_to_iso3[lang],
                        "Score": score,
                        "Metric": metrics[metric],
                    }
                )

pd.DataFrame.from_records(results).to_csv("length-data.csv", index=False)
