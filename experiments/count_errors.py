import argparse

import editdistance
import pandas as pd

from src.dataset.prepare_dataset import split_interleaved_segments

parser = argparse.ArgumentParser()
parser.add_argument("csv_path")
args = parser.parse_args()

df = pd.read_csv(args.csv_path)

hallucinated_transcript = 0
avg_edit_dist = 0
bad_format = 0
repetition = 0

for _, row in df.iterrows():
    if len(row["predicted"]) > 20000:
        repetition += 1
        continue
    if "(" not in row["predicted"] or ")" not in row["predicted"]:
        bad_format += 1
        continue
    pred_seg, pred_glos = split_interleaved_segments(row["predicted"].strip())  # type:ignore
    gold_seg, gold_glos = split_interleaved_segments(row["reference"].strip())  # type:ignore
    pred_seg = pred_seg.replace("-", "").replace("=", "")
    gold_seg = gold_seg.replace("-", "").replace("=", "")

    avg_edit_dist += editdistance.eval(pred_seg, gold_seg) / len(gold_seg)

    if pred_seg != gold_seg:
        hallucinated_transcript += 1
        continue

avg_edit_dist /= len(df) - bad_format - repetition

print(
    f"""Hallucinated: {hallucinated_transcript / len(df):%}
    Bad format: {bad_format / len(df):%}
    Repetition: {repetition / len(df):%}
    Avg edit dist: {avg_edit_dist:.2f}"""
)
