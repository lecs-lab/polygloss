import wandb
from data.scrape_data import evaluation_languages

api = wandb.Api()

metrics = ["glossing.morphemes.error_rate", "segmentation.f1", "alignment"]
runs = [
    "polygloss-byt5-multitask (fixed eval)",
    "polygloss-byt5-concat (fixed eval)",
    "polygloss-byt5-interleaved (fixed eval)",
]
nice_names = [
    "\\textsc{PolyGloss} (ByT5, multitask)",
    "\\textsc{PolyGloss} (ByT5, concat)",
    "\\textsc{PolyGloss} (ByT5, interleaved)",
]

results = {run_name: {m: [] for m in metrics} for run_name in runs}

for run in api.runs(path="lecs-general/polygloss"):
    if run.name in results:
        for metric in results[run.name]:
            for lang in evaluation_languages:
                score = run.summary_metrics["test"][lang]
                for key in metric.split("."):
                    score = score[key]
                results[run.name][metric].append(score)
            average = sum(results[run.name][metric]) / len(results[run.name][metric])
            results[run.name][metric].append(average)

# Print LaTeX
for m in metrics:
    print(f"========={m}=========")
    for run_name, nice_name in zip(runs, nice_names):
        print(
            "    "
            + nice_name
            + " & "
            + " & ".join([f"{r:.3f}" for r in results[run_name][m]])
            + " \\\\"
        )
