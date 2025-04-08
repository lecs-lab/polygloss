"""Defines models and functions for loading, manipulating, and writing task data"""

import pathlib
import re
from dataclasses import dataclass
from typing import List, Literal, Optional

SplitType = Literal["train", "dev", "test"]


@dataclass
class IGTLine:
    """A single instance of IGT in some language."""

    id: str
    source: str

    transcription: str
    segmentation: Optional[str]
    glosses: Optional[str]
    translation: Optional[str] = None

    glottocode: Optional[str] = None
    metalang_glottocode: Optional[str] = None
    designated_split: Optional[SplitType] = None

    def __repr__(self):
        return f"Trnsc:\t{self.transcription}\nSegm:\t{self.segmentation}\nGloss:\t{self.glosses}\nTrnsl:\t{self.translation}\n\n"

    def gloss_list(self, segmented=False) -> Optional[List[str]]:
        """Returns the gloss line of the IGT as a list.
        :param segmented: If True, will return each morpheme gloss as a separate item.
        """
        if self.glosses is None:
            return []
        if not segmented:
            return self.glosses.split()
        else:
            return re.split(r"\s|-", self.glosses)


def load_igt(path: pathlib.Path, id_prefix: str, source: str):
    """Loads a file containing IGT data into a list of entries.

    Args:
        path (pathlib.Path): The path to the file or directory to load.
        id_prefix (str): A prefix to use for the ID of each entry.
        source (str): The source of the data, used for metadata.
    """
    all_data: list[IGTLine] = []

    # If we have a directory, recursively load all files and concat together
    if path.is_dir():
        for file in path.iterdir():
            if file.suffix == ".txt":
                print(file)
                all_data.extend(load_igt(file, id_prefix=id_prefix, source=source))
        return all_data

    # If we have one file, read in line by line
    with open(path, "r") as file:
        current_entry: dict[str, str] = {}
        skipped_lines = []
        counter = 0  # Used to assign incrementing IDs

        for line in file:
            # Determine the type of line
            # If we see a type that has already been filled for the current entry, something is wrong
            line_prefix = line[:2]
            if line_prefix == "\\t" and current_entry.get("transcription") is None:
                current_entry["transcription"] = line[3:].strip()
            elif line_prefix == "\\m" and current_entry.get("segmentation") is None:
                current_entry["segmentation"] = line[3:].strip()
            elif line_prefix == "\\g" and current_entry.get("glosses") is None:
                if len(line[3:].strip()) > 0:
                    current_entry["glosses"] = line[3:].strip()
            elif line_prefix == "\\l" and current_entry.get("translation") is None:
                current_entry["translation"] = line[3:].strip()

                # Once we have the translation, we've reached the end and can save this entry
                if current_entry.get("transcription") is None:
                    skipped_lines.append(line)
                else:
                    all_data.append(
                        IGTLine(
                            id=f"{id_prefix}_{counter}",
                            source=source,
                            **current_entry,  # type:ignore
                        )
                    )
                    counter += 1
                current_entry = {}
            elif line_prefix == "\\p":
                # Skip POS lines
                continue
            elif line.strip() != "":
                # Something went wrong
                skipped_lines.append(line)
                continue
            else:
                if (
                    len(current_entry) > 0
                    and current_entry.get("transcription") is not None
                ):
                    all_data.append(
                        IGTLine(
                            id=f"{id_prefix}_{counter}",
                            source=source,
                            **current_entry,  # type:ignore
                        )
                    )
                    counter += 1
                    current_entry = {}

        # Might have one extra line at the end
        if len(current_entry) > 0 and current_entry.get("transcription") is not None:
            all_data.append(
                IGTLine(
                    id=f"{id_prefix}_{counter}",
                    source=source,
                    **current_entry,  # type:ignore
                )
            )
        if len(skipped_lines) != 0:
            print(f"Skipped {len(skipped_lines)} lines")
            print(skipped_lines)
    return all_data
