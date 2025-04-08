import pathlib
import typing

from tqdm import tqdm

from data.data import IGTLine, SplitType, load_igt


def scrape_odin() -> list[IGTLine]:
    raw_dir = pathlib.Path(__file__).parent / "raw/ODIN"
    all_data: list[IGTLine] = []
    for file in tqdm((raw_dir / "odin_data_sigmorphon").iterdir()):
        data = load_igt(file, id_prefix=f"odin_{file.stem}", source="odin")
        for row in data:
            # TODO: use mapping from odin codes to glottocodes
            pass
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
                        split = s
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
                    row.designated_split = typing.cast(SplitType, split)
                all_data.extend(data)

    return all_data
