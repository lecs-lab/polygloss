import subprocess
import sys
from pathlib import Path

import wandb


def log_pip_freeze_artifact(name="pip-freeze"):
    out = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
    )

    path = Path("pip_freeze.txt")
    path.write_text(out)

    artifact = wandb.Artifact(
        name=name,
        type="environment",
        description="Exact pip freeze of runtime environment",
    )
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)
