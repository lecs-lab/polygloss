import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

import wandb
from data.scrape_data import evaluation_languages

api = wandb.Api()

RUN_ID = "nuiz8jqs"
ARTIFACT_ID = "run-nuiz8jqs-evalloss_per_language-afc56d0b2a0a7996dffc6f3107d989f1:v0"

run = api.run(f"lecs-general/polygloss/{RUN_ID}")
artifact = api.artifact(f"lecs-general/polygloss/{ARTIFACT_ID}")
dir = artifact.download()
with open(f"{dir}/eval/loss_per_language.table.json", "r") as f:
    json_dict = json.load(f)
df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])

viz_dir = Path(__file__).parent / "viz"
viz_dir.mkdir(exist_ok=True)

sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(4, 2.5))

# Histogram of ppls
sns.histplot(data=df[df["ppl"] < 5.0], x="ppl", ax=ax)
plt.show()
plt.close(f)

# Scatter plot
metrics = ["glossing.morphemes.error_rate", "segmentation.f1", "alignment"]
rows = []
for lang in evaluation_languages:
    row = {"Lang": lang}
    for metric in metrics:
        score = run.summary_metrics["test"][lang]
        for key in metric.split("."):
            score = score[key]
        row[metric] = score
    row["PPL"] = df[df["glottocode"] == lang]["ppl"].item()
    rows.append(row)

scores = pd.DataFrame(rows)

X_METRIC = "PPL"
Y_METRIC = "glossing.morphemes.error_rate"
f, ax = plt.subplots(figsize=(4, 2.5))
scatter = sns.scatterplot(data=scores, x=X_METRIC, y=Y_METRIC, hue="Lang", ax=ax)
sns.regplot(
    data=scores,
    x=X_METRIC,
    y=Y_METRIC,
    scatter=False,
    ax=ax,
    ci=None,
    line_kws={"color": "grey", "linestyle": "--"},
)
plt.legend(fontsize="xx-small", loc="lower right")
plt.xlabel("Perplexity")
plt.ylabel("Glossing Error Rate")
plt.savefig(
    viz_dir / "corr.pdf", format="pdf", bbox_inches="tight"
)  # bbox_inches="tight" removes extra whitespace
plt.show()

r2 = stats.pearsonr(scores["PPL"], scores["glossing.morphemes.error_rate"])[0] ** 2
print(f"r^2={r2}")
