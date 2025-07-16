"""Contains individual functions for scraping each of the original datasets"""

import pathlib
import typing

import datasets
import pandas as pd
from tqdm import tqdm

from data.lang_codes import iso1_to_3, iso3_to_glotto, odin_to_glotto
from data.model import IGTLine, load_igt

in_domain_glottocodes = ["arap1274", "dido1241", "uspa1245", "ainu1240"]
out_of_domain_glottocodes = ["lezg1247", "natu1246", "nyan1302", "ruul1235"]


def scrape_odin() -> list[IGTLine]:
    raw_dir = pathlib.Path(__file__).parent / "raw/ODIN"
    all_data: list[IGTLine] = []
    for file in tqdm((raw_dir / "odin_data_sigmorphon").iterdir()):
        if file.stem == "TOTAL":
            continue
        glottocode = odin_to_glotto.get(file.stem.split("-")[0])
        data = load_igt(file, id_prefix=f"odin_{file.stem}", source="odin")
        for row in data:
            row.glottocode = glottocode
        all_data.extend(data)
    return all_data


def scrape_sigmorphon_st() -> list[IGTLine]:
    lang_mapping = {
        "Arapaho": ("arap1274", "stan1293"),
        "Gitksan": ("gitx1241", "stan1293"),
        "Lezgi": ("lezg1247", "stan1293"),
        "Natugu": ("natu1246", "stan1293"),
        "Nyangbo": ("nyan1302", "stan1293"),
        "Tsez": ("dido1241", "stan1293"),
        "Uspanteko": ("uspa1245", "stan1288"),
    }

    raw_dir = pathlib.Path(__file__).parent / "raw/sigmorphon_st"
    all_data: list[IGTLine] = []
    for lang_folder in raw_dir.iterdir():
        if not lang_folder.is_dir():
            print(f"Skipping {lang_folder}, not a directory!")
            continue
        glottocode, metalang_code = lang_mapping[lang_folder.name]
        for file in lang_folder.iterdir():
            if "track2-uncovered" in file.name:
                for s in ["train", "dev", "test"]:
                    if s in file.name:
                        # Maintain the splits as the original dataset
                        if glottocode in in_domain_glottocodes and s != "train":
                            split = f"{s}_ID"
                        elif glottocode in out_of_domain_glottocodes:
                            split = f"{s}_OOD"
                        else:
                            split = None
                        break
                else:
                    raise ValueError("File name must contain split: ", file)

                data = load_igt(
                    file,
                    id_prefix=f"sigmorphon_st_{glottocode}",
                    source="sigmorphon_st",
                )
                for row in data:
                    row.glottocode = glottocode
                    row.metalang_glottocode = metalang_code
                    row.designated_split = split
                all_data.extend(data)

    return all_data


def scrape_cldf() -> list[IGTLine]:
    data: list[IGTLine] = []

    for dataset in ["apics", "uratyp"]:
        raw_dir = pathlib.Path(__file__).parent / f"raw/{dataset}/cldf/"
        df = pd.read_csv(raw_dir / "examples.csv")
        langs_df = pd.read_csv(raw_dir / "languages.csv")
        df = pd.merge(df, langs_df, left_on="Language_ID", right_on="ID", how="left")
        df = df[["Analyzed_Word", "Gloss", "Translated_Text", "Glottocode", "ID_x"]]
        df = df[pd.notnull(df["Analyzed_Word"])]
        df = df[pd.notnull(df["Gloss"])]
        df = typing.cast(pd.DataFrame, df)

        for row in df.itertuples():
            row = typing.cast(typing.Any, row)
            transcription = row.Analyzed_Word.replace("\\t", " ")
            segmentation = transcription if "-" in transcription else None
            transcription = transcription.replace("-", "")
            glottocode = row.Glottocode if isinstance(row.Glottocode, str) else None
            data.append(
                IGTLine(
                    id=f"{dataset}_{row.ID_x}",
                    source=dataset,
                    transcription=transcription,
                    segmentation=segmentation,
                    glosses=row.Gloss.replace("\\t", " "),
                    translation=row.Translated_Text,
                    glottocode=glottocode,
                )
            )
    return data


def scrape_imtvault() -> list[IGTLine]:
    raw_dir = pathlib.Path(__file__).parent / "raw/imtvault/cldf"
    df = pd.read_csv(raw_dir / "examples.csv")
    data: list[IGTLine] = []
    for row in df.itertuples():
        row = typing.cast(typing.Any, row)
        if isinstance(row.Analyzed_Word, float) or isinstance(row.Gloss, float):
            continue
        transcription = row.Analyzed_Word.replace("\\t", " ")
        segmentation = transcription if "-" in transcription else None
        transcription = transcription.replace("-", "")
        glottocode = row.Language_ID if isinstance(row.Language_ID, str) else None
        metalang_glottocode = (
            row.Meta_Language_ID if isinstance(row.Meta_Language_ID, str) else None
        )
        data.append(
            IGTLine(
                id=f"imtvault_{row.ID}",
                source="imtvault",
                transcription=transcription,
                segmentation=segmentation,
                glosses=row.Gloss.replace("\\t", " "),
                translation=row.Translated_Text,
                glottocode=glottocode,
                metalang_glottocode=metalang_glottocode,
            )
        )
    return data


def scrape_gurani() -> list[IGTLine]:
    raw_dir = pathlib.Path(__file__).parent / "raw/guarani/data-fixed"
    all_data: list[IGTLine] = []
    for file in raw_dir.iterdir():
        if not file.suffix == ".txt":
            print(f"Skipping {file}, not a text file!")
            continue
        data = load_igt(
            file,
            id_prefix="guarani",
            source="guarani",
        )
        all_data.extend(data)
    return all_data


def scrape_fieldwork() -> list[IGTLine]:
    dataset = datasets.load_dataset(
        "wav2gloss/fieldwork",
        streaming=True,
        columns=[
            "id",
            "translation_language",
            "transcription",
            "surface",
            "gloss",
            "translation",
            "language",
        ],
    )
    dataset = typing.cast(datasets.DatasetDict, dataset)

    def _remove_spaces(s: str):
        return (
            s.replace("- ", "-")
            .replace(" -", "-")
            .replace("= ", "=")
            .replace(" =", "=")
        )

    all_data: list[IGTLine] = []
    for split in dataset.keys():
        for row in tqdm(dataset[split]):
            row = typing.cast(typing.Mapping, row)
            glottocode = row["language"]
            iso1 = row["translation_language"]
            iso1 = typing.cast(str, {"sh": "hr", "eml": "egl"}.get(iso1, iso1))
            metalang_glottocode = iso3_to_glotto.get(iso1_to_3(iso1))

            if glottocode in in_domain_glottocodes and split != "train":
                designated_split = f"{split if split != 'validation' else 'dev'}_ID"
            elif glottocode in out_of_domain_glottocodes:
                designated_split = f"{split if split != 'validation' else 'dev'}_OOD"
            else:
                designated_split = None

            all_data.append(
                IGTLine(
                    id=row["id"],
                    source="wav2gloss",
                    transcription=row["transcription"],
                    segmentation=_remove_spaces(row["surface"]),
                    glosses=_remove_spaces(row["gloss"]),
                    translation=row["translation"],
                    glottocode=glottocode,
                    metalang_glottocode=metalang_glottocode,
                    designated_split=designated_split,
                )
            )
    return all_data
