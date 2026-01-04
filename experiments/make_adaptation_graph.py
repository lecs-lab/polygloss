import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(4, 1.5))
df = pd.read_csv("experiments/adaptation-data.csv")
sns.lineplot(
    df,
    x="size",
    y="mer",
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
# ax.set_ylim(0, 1)
plt.savefig("experiments/viz/adaptation.pdf", format="pdf", bbox_inches="tight")
plt.show()
