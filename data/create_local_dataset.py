"""Create a local version of the data for one language for easy analysis"""

import argparse
import pathlib
import typing

import datasets

parser = argparse.ArgumentParser()
parser.add_argument("glottocode", help="Glottocode for the language to filter by")
parser.add_argument("--output-dir", help="Path to output files to", default="./local")
parser.add_argument("--dataset", default="lecslab/polygloss-corpus")
args = parser.parse_args()

dataset = datasets.load_dataset(args.dataset)
dataset = typing.cast(datasets.DatasetDict, dataset)
dataset = dataset.filter(lambda row: row["glottocode"] == args.glottocode)

output_folder = pathlib.Path(args.output_dir) / args.glottocode
output_folder.mkdir(parents=True, exist_ok=True)

for split in dataset:
    if len(dataset[split]) > 0:
        output_path = output_folder / f"{split}.igt"
        with open(output_path, "w") as f:
            for row in dataset[split]:
                row = typing.cast(typing.Mapping, row)
                f.write("\\t " + row["transcription"] + "\n")
                if row["segmentation"] and len(row["segmentation"]) > 0:
                    f.write("\\m " + row["segmentation"] + "\n")
                f.write("\\g " + row["glosses"] + "\n")
                if row["translation"] and len(row["translation"]) > 0:
                    f.write("\\l " + row["translation"] + "\n")
                f.write("\n")

print(f"Wrote files to {output_folder}")
