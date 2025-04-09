import csv
import json
import pathlib

from iso639 import Lang

glotto_to_iso3: dict[str, str] = {}
iso3_to_glotto: dict[str, str] = {}

with open(pathlib.Path(__file__).parent / "raw/glottocode2iso.csv", "r") as f:
    csvreader = csv.reader(f)
    header = next(csvreader)
    for row in csvreader:
        [glottocode, isocode] = row
        glotto_to_iso3[glottocode] = isocode
        iso3_to_glotto[isocode] = glottocode


def iso1_to_3(code: str):
    """Converts ISO 639-1 codes (two-letter) into ISO 639-3 (three-letter)"""
    return Lang(code).pt3


with open(pathlib.Path(__file__).parent / "raw/ODIN/language_map.json", "r") as f:
    odin_to_glotto: dict[str, str] = json.load(f)
