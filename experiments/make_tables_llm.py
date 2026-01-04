import wandb
from data.scrape_data import evaluation_languages

api = wandb.Api()

metrics = ["glossing.morphemes.error_rate", "segmentation.f1", "alignment"]
models = ["Qwen/Qwen3-0.6B", "CohereLabs/aya-expanse-8b", "google/gemma-3-4b-it"]
nice_names = ["Qwen 3 (ICL)", "Aya Expanse (ICL)", "Gemma 3 (ICL)"]

results = {
    run_name: {m: [None for _ in range(len(evaluation_languages))] for m in metrics}
    for run_name in models
}

for run in api.runs(path="lecs-general/polygloss.llm-baseline"):
    model = run.config["model"]
    glotto = run.config["glottocode"]
    index = evaluation_languages.index(glotto)
    for metric in metrics:
        if "test" not in run.summary_metrics:
            continue
        score = run.summary_metrics["test"][glotto]
        for key in metric.split("."):
            if key not in score:
                score = None
                break
            score = score[key]
        results[model][metric][index] = score


for model, model_scores in results.items():
    for metric, scores in model_scores.items():
        if all(m is not None for m in scores):
            average = sum(scores) / len(scores)
        else:
            average = None
        results[model][metric].append(average)

# Print LaTeX
for m in metrics:
    print(f"========={m}=========")
    for run_name, nice_name in zip(models, nice_names):
        print(
            "    "
            + nice_name
            + " & "
            + " & ".join(
                [f"{r:.3f}" if r is not None else "" for r in results[run_name][m]]
            )
            + " \\\\"
        )
