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
f, ax = plt.subplots(figsize=(4, 2))

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
scores.to_csv("test-scores.csv")

X_METRIC = "PPL"
Y_METRIC = "glossing.morphemes.error_rate"
f, ax = plt.subplots(figsize=(4, 1.5))
scatter = sns.scatterplot(data=scores, x=X_METRIC, y=Y_METRIC, hue="Lang", ax=ax)
sns.regplot(
    data=scores,
    x=X_METRIC,
    y=Y_METRIC,
    scatter=False,
    ax=ax,
    ci=None,
    line_kws={"color": "black", "linestyle": "--", "linewidth": 1},
)
leg = ax.legend(
    loc="lower center",
    fontsize="x-small",
    bbox_to_anchor=(0.5, 1.02),
    bbox_transform=ax.transAxes,
    ncol=5,
    frameon=False,
    title=None,
    borderaxespad=0.0,
    columnspacing=1.2,
    handletextpad=0.6,
)
# leg = plt.legend(fontsize="x-small", loc="lower right")
for text in leg.get_texts():
    text.set_fontweight("bold")
plt.xlabel("Perplexity", fontweight="bold")
plt.ylabel("MER", fontweight="bold")
plt.savefig(
    viz_dir / "corr.pdf", format="pdf", bbox_inches="tight"
)  # bbox_inches="tight" removes extra whitespace
plt.show()

regressions = {}

for metric in metrics:
    print("\n" + metric)
    r2 = stats.pearsonr(scores["PPL"], scores[metric])[0] ** 2
    print(f"r^2={r2}")
    reg = stats.linregress(scores["PPL"], scores[metric])
    regressions[metric] = {"slope": reg[0], "intercept": reg[1]}
    print(reg)

pd.DataFrame.from_dict(regressions).transpose().to_csv("regressions.csv")
