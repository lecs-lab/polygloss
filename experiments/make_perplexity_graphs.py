import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

import wandb
from data.scrape_data import evaluation_isocodes, evaluation_languages

api = wandb.Api()

RUN_ID = "i4nlgun4"
ARTIFACT_ID = "run-i4nlgun4-evalloss_per_language-659c5b48426ce30e35b6d6fdec0e6d26:v0"

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
for lang, iso in zip(evaluation_languages, evaluation_isocodes):
    row = {"Lang": iso}
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
leg = plt.legend(fontsize="x-small", loc="lower right")
for text in leg.get_texts():
    text.set_fontweight("bold")
plt.xlabel("Perplexity", fontweight="bold")
plt.ylabel("Glossing Error Rate", fontweight="bold")
plt.savefig(
    viz_dir / "corr.pdf", format="pdf", bbox_inches="tight"
)  # bbox_inches="tight" removes extra whitespace
plt.show()

r2 = stats.pearsonr(scores["PPL"], scores["glossing.morphemes.error_rate"])[0] ** 2
print(f"r^2={r2}")
