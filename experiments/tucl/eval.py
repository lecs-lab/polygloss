from pathlib import Path

import pandas as pd

import wandb
from data.model import load_igt
from data.scrape_data import evaluation_isocodes, evaluation_languages
from src.evaluation.evaluate import evaluate

wandb.init(project="polygloss", entity="lecs-general", name="TU-CL")

all_preds = []

for glotto, iso in zip(evaluation_languages, evaluation_isocodes):
    try:
        preds = load_igt(
            Path(__file__).parent / "pred" / f"{iso}_track1_morph_trial1.prediction",
            id_prefix=iso,
            source="sigmorphon_st",
        )
        gold = load_igt(
            Path(__file__).parent / "gold" / f"{iso}-test-track2-uncovered",
            id_prefix=iso,
            source="sigmorphon_st",
        )
    except:
        continue
    for p, g in zip(preds, gold):
        all_preds.append(
            {
                "predicted": p.glosses,
                "reference": g.glosses,
                "task": "t2g",
                "id": p.id,
                "glottocode": glotto,
            }
        )
        all_preds.append(
            {
                "predicted": p.segmentation,
                "reference": g.segmentation,
                "task": "t2s",
                "id": p.id,
                "glottocode": glotto,
            }
        )

df = pd.DataFrame(all_preds)
wandb.log({"predictions": wandb.Table(dataframe=df)})
metrics = evaluate(df)
wandb.log(data={"test": metrics})
