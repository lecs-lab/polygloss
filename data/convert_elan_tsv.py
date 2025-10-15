#!/usr/bin/env python3
"""
convert_chunked_csv_to_toolbox_pandas.py

Usage:
    python convert_chunked_csv_to_toolbox_pandas.py INPUT_DIR OUTPUT_PREFIX \
        --transcription-col Transcription \
        --segmentation-col SurfaceSegmentation \
        --gloss-col SurfaceGloss \
        --translation-col Translation

Description:
    Reads CSV files in INPUT_DIR, chunks by non-empty transcription rows,
    reconstructs segmentation and gloss lines from each chunk,
    and writes Toolbox-format train/dev/test files.
"""

import pandas as pd
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional


ENCODINGS_TO_TRY = ["utf-8", "utf-16", "cp1252"]


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Try multiple encodings until one works."""
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            continue
    raise UnicodeDecodeError(f"Could not decode file {path} with encodings {ENCODINGS_TO_TRY}")


def chunk_dataframe(df: pd.DataFrame, transcription_col: str) -> List[pd.DataFrame]:
    """Split the dataframe into chunks where a non-empty transcription starts a new chunk."""
    chunks = []
    current_rows = []
    for _, row in df.iterrows():
        transcription = str(row.get(transcription_col, "")).strip()
        if transcription and transcription.lower() != "nan":
            # Start of a new chunk
            if current_rows:
                chunks.append(pd.DataFrame(current_rows))
            current_rows = [row]
        else:
            if current_rows:
                current_rows.append(row)
    if current_rows:
        chunks.append(pd.DataFrame(current_rows))
    return chunks


def reconstruct_entry(chunk: pd.DataFrame, cols: Dict[str, str]) -> Dict[str, str]:
    """Create a Toolbox-style entry from a chunk."""
    transcription = str(chunk.iloc[0].get(cols["transcription"], "")).strip()

    segs = [str(x).strip() for x in chunk[cols["segmentation"]].dropna() if str(x).strip()]
    glosses = [str(x).strip() for x in chunk[cols["gloss"]].dropna() if str(x).strip()]

    translation = ""
    if cols.get("translation") in chunk.columns:
        translation = str(chunk.iloc[0].get(cols["translation"], "")).strip()

    return {
        "transcription": transcription,
        "segmentation": " ".join(segs),
        "glosses": " ".join(glosses),
        "translation": translation,
    }


def write_toolbox(entries: List[Dict[str, str]], out_path: Path):
    """Write list of entries to Toolbox format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(f"\\t {e['transcription']}\n")
            f.write(f"\\m {e['segmentation']}\n")
            f.write(f"\\g {e['glosses']}\n")
            f.write(f"\\l {e['translation']}\n\n")


def split_dataset(entries: List[Dict[str, str]], train_ratio=0.8, dev_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(entries)
    n = len(entries)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    return entries[:train_end], entries[train_end:dev_end], entries[dev_end:]


def process_directory(input_dir: str, output_prefix: str, cols: Dict[str, str]):
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    all_entries = []
    for file in csv_files:
        df = read_csv_with_fallback(file)
        if cols["transcription"] not in df.columns:
            raise KeyError(f"Missing column '{cols['transcription']}' in {file}")
        chunks = chunk_dataframe(df, cols["transcription"])
        for chunk in chunks:
            e = reconstruct_entry(chunk, cols)
            if e["transcription"]:  # only add if transcription present
                all_entries.append(e)

    print(f"✅ Parsed {len(all_entries)} utterances from {len(csv_files)} files")

    train, dev, test = split_dataset(all_entries)
    print(f"Split into train={len(train)}, dev={len(dev)}, test={len(test)}")

    write_toolbox(train, Path(f"{output_prefix}-train.txt"))
    write_toolbox(dev, Path(f"{output_prefix}-dev.txt"))
    write_toolbox(test, Path(f"{output_prefix}-test.txt"))
    print(f"✅ Toolbox files written to '{output_prefix}-*.txt'")


def main():
    parser = argparse.ArgumentParser(description="Convert elan csvs into Toolbox-format datasets.")
    parser.add_argument("--input_dir", default="/projects/enri8153/polygloss/data/zong1243/raw",  help="Directory with CSV files")
    parser.add_argument("--output_prefix", default="/projects/enri8153/polygloss/data/zong1243/parsed/zong1243", help="Prefix for output files")
    parser.add_argument("--transcription-col", default="Transcription")
    parser.add_argument("--segmentation-col", default="Surface Segmentation")
    parser.add_argument("--gloss-col", default="Surface Glossing")
    parser.add_argument("--translation-col", default="Translation")
    args = parser.parse_args()

    cols = {
        "transcription": args.transcription_col,
        "segmentation": args.segmentation_col,
        "gloss": args.gloss_col,
        "translation": args.translation_col,
    }

    process_directory(args.input_dir, args.output_prefix, cols)


if __name__ == "__main__":
    main()
