import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(4, 1.5))
df = pd.read_csv("experiments/adaptation-data.csv")
df = df.melt(
    id_vars="size",
    value_vars=["PolyGloss", "ByT5"],
    value_name="MER",
    var_name="Base Model",
)
sns.lineplot(
    df,
    x="size",
    y="MER",
    hue="Base Model",
    ax=ax,
)
plt.xlabel("Train Size", fontweight="bold")
plt.ylabel("MER", fontweight="bold")
ax.tick_params(
    axis="both",
    which="both",
    bottom=True,
    top=False,
    left=True,
    length=4,
    width=1,
    labelsize=9,
)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
leg = ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),  # x=0 is the *axes* left edge
    bbox_transform=ax.transAxes,  # <-- crucial
    ncol=2,
    frameon=False,
    title=None,
    borderaxespad=0.0,
    columnspacing=1.2,
    handletextpad=0.6,
)
for text in leg.get_texts():
    text.set_fontweight("bold")
# ax.set_ylim(0, 1)
plt.savefig("experiments/viz/adaptation.pdf", format="pdf", bbox_inches="tight")
plt.show()
