import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")

# Ablation 1
f, ax = plt.subplots(figsize=(4, 3))
df = pd.read_csv("experiments/ablation-data.csv")
df["Metric"] = df["Metric"].replace(
    {
        "Segmentation F1": "Segm.\n[F1]",
        "Glossing MER": "Gloss\n[MER]",
        "Alignment": "Align.",
    }
)
data = df[df["Method"].isin(["Separate", "Joint", "Pipeline", "Girrbach (2023)"])]
# summary = df.groupby(["Method", "Metric"], as_index=False).agg(
#     Mean=("Score", "mean"),
#     sem=("Score", "sem"),
# )
ax.tick_params(
    axis="x",
    which="both",
    bottom=True,
    top=False,
    length=4,
    width=1,
    labelsize=9,
)
g = sns.barplot(
    data=data,
    x="Score",
    y="Metric",
    hue="Method",
    palette="deep",
    edgecolor="black",
    estimator="mean",
    errorbar="se",
    linewidth=1,
    alpha=1,
    ax=ax,
    # legend=None,
)
leg = ax.legend(
    loc="lower left",
    bbox_to_anchor=(0, 1.02),  # x=0 is the *axes* left edge
    bbox_transform=ax.transAxes,  # <-- crucial
    ncol=2,
    frameon=False,
    title=None,
    borderaxespad=0.0,
    columnspacing=1.2,
    handletextpad=0.6,
)

# Make the legend columns left-aligned internally
leg._legend_box.align = "left"
sns.despine(
    bottom=False,
)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.set_xlabel("Mean Score", fontweight="bold")
ax.set(ylabel=None)
ax.set_xlim(0, 1)
for text in ax.legend_.get_texts():
    text.set_fontweight("bold")
plt.savefig(
    "experiments/viz/ablation1.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01
)
plt.show()
plt.close(f)


# Ablation 2
f, ax = plt.subplots(figsize=(4, 2))
data = df[df["Method"].isin(["Joint", "PolyGloss [interleaved]"])]
data["Method"] = data["Method"].replace(
    {"Joint": "Monolingual", "PolyGloss [interleaved]": "PolyGloss"}
)
# summary = df.groupby(["Method", "Metric"], as_index=False).agg(
#     Mean=("Score", "mean"),
#     sem=("Score", "sem"),
# )
ax.tick_params(
    axis="x",
    which="both",
    bottom=True,
    top=False,
    length=4,
    width=1,
    labelsize=9,
)
g = sns.barplot(
    data=data,
    x="Score",
    y="Metric",
    hue="Method",
    palette="deep",
    edgecolor="black",
    estimator="mean",
    errorbar="se",
    linewidth=1,
    alpha=1,
    ax=ax,
    # legend=None,
)
leg = ax.legend(
    loc="lower left",
    bbox_to_anchor=(0, 1.02),  # x=0 is the *axes* left edge
    bbox_transform=ax.transAxes,  # <-- crucial
    ncol=2,
    frameon=False,
    title=None,
    borderaxespad=0.0,
    columnspacing=1.2,
    handletextpad=0.6,
)

# Make the legend columns left-aligned internally
leg._legend_box.align = "left"
sns.despine(
    bottom=False,
)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.set_xlabel("Mean Score", fontweight="bold")
ax.set(ylabel=None)
ax.set_xlim(0, 1)
for text in ax.legend_.get_texts():
    text.set_fontweight("bold")
plt.savefig(
    "experiments/viz/ablation2.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01
)
plt.show()
plt.close(f)


# Ablation 3
f, ax = plt.subplots(figsize=(4, 3))
data = df[
    df["Method"].isin(
        ["PolyGloss [multitask]", "PolyGloss [interleaved]", "PolyGloss [concat]"]
    )
]
data["Method"] = data["Method"].replace(
    {
        "PolyGloss [interleaved]": "Interleaved",
        "PolyGloss [concat]": "Concat",
        "PolyGloss [multitask]": "Multitask",
    }
)
# summary = df.groupby(["Method", "Metric"], as_index=False).agg(
#     Mean=("Score", "mean"),
#     sem=("Score", "sem"),
# )
ax.tick_params(
    axis="x",
    which="both",
    bottom=True,
    top=False,
    length=4,
    width=1,
    labelsize=9,
)
g = sns.barplot(
    data=data,
    x="Score",
    y="Metric",
    hue="Method",
    palette="deep",
    edgecolor="black",
    estimator="mean",
    errorbar="se",
    linewidth=1,
    alpha=1,
    ax=ax,
    # legend=None,
)
leg = ax.legend(
    loc="lower left",
    bbox_to_anchor=(0, 1.02),  # x=0 is the *axes* left edge
    bbox_transform=ax.transAxes,  # <-- crucial
    ncol=2,
    frameon=False,
    title=None,
    borderaxespad=0.0,
    columnspacing=1.2,
    handletextpad=0.6,
)

# Make the legend columns left-aligned internally
leg._legend_box.align = "left"
sns.despine(
    bottom=False,
)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.set_xlabel("Mean Score", fontweight="bold")
ax.set(ylabel=None)
ax.set_xlim(0, 1)
for text in ax.legend_.get_texts():
    text.set_fontweight("bold")
plt.savefig(
    "experiments/viz/ablation3.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01
)
plt.show()
plt.close(f)
