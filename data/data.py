"""Defines models and functions for loading, manipulating, and writing task data"""
from typing import Optional, List
import re
import os
import datasets

class IGTLine:
    """A single line of IGT"""
    def __init__(self, transcription: str, segmentation: Optional[str], glosses: Optional[str], translation: Optional[str]):
        self.transcription = transcription
        self.segmentation = segmentation
        self.glosses = glosses
        self.translation = translation
        self.should_segment = True

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
            return re.split("\s|-", self.glosses)

    def __dict__(self):
        d = {'transcription': self.transcription, 'translation': self.translation}
        if self.glosses is not None:
            d['glosses'] = self.gloss_list(segmented=self.should_segment)
        if self.segmentation is not None:
            d['segmentation'] = self.segmentation
        return d

    
def load_data_file(path: str):
    """Loads a file containing IGT data into a list of entries."""
    all_data = []

    # If we have a directory, recursively load all files and concat together
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(".txt"):
                print(file)
                all_data.extend(load_data_file(os.path.join(path, file)))
        return all_data

    # If we have one file, read in line by line
    with open(path, 'r') as file:
        current_entry = [None, None, None, None]  # transc, segm, gloss, transl

        skipped_lines = []
        
        for line in file:
            # Determine the type of line
            # If we see a type that has already been filled for the current entry, something is wrong
            line_prefix = line[:2]
            if line_prefix == '\\t' and current_entry[0] == None:
                current_entry[0] = line[3:].strip()
            elif line_prefix == '\\m' and current_entry[1] == None:
                current_entry[1] = line[3:].strip()
            elif line_prefix == '\\g' and current_entry[2] == None:
                if len(line[3:].strip()) > 0:
                    current_entry[2] = line[3:].strip()
            elif line_prefix == '\\l' and current_entry[3] == None:
                current_entry[3] = line[3:].strip()
                # Once we have the translation, we've reached the end and can save this entry
                all_data.append(IGTLine(transcription=current_entry[0],
                                        segmentation=current_entry[1],
                                        glosses=current_entry[2],
                                        translation=current_entry[3]))
                current_entry = [None, None, None, None]
            elif line_prefix == "\\p":
                # Skip POS lines
                continue
            elif line.strip() != "":
                # Something went wrong
                skipped_lines.append(line)
                continue
            else:
                if not current_entry == [None, None, None, None]:
                    all_data.append(IGTLine(transcription=current_entry[0],
                                            segmentation=current_entry[1],
                                            glosses=current_entry[2],
                                            translation=None))
                    current_entry = [None, None, None, None]
        # Might have one extra line at the end
        if not current_entry == [None, None, None, None]:
            all_data.append({"transcr"})
            all_data.append(IGTLine(transcription=current_entry[0],
                                    segmentation=current_entry[1],
                                    glosses=current_entry[2],
                                    translation=None))
        if len(skipped_lines) == 0:
            print("Looks good")
        else:
            print(f"Skipped {len(skipped_lines)} lines")
            print(skipped_lines)
    return all_data
        
        
def create_hf_dataset(filename, glottocode, metalang, row_id='st'):
    print(f"Loading {filename}")
    raw_data = load_data_file(filename)
    data = []
    for i, line in enumerate(raw_data):
        new_row = {'glottocode': glottocode, 'metalang_glottocode': metalang, "is_segmented": "yes", "source": "sigmorphon_st", "type": "canonical"}
        new_row['ID'] = f"{row_id}_{glottocode}_{i}"
        new_row['transcription'] = line.segmentation
        new_row['glosses'] = line.glosses
        new_row['translation'] = line.translation
        data.append(new_row)

        new_row_unsegmented = {'glottocode': glottocode, 'metalang_glottocode': metalang, "is_segmented": "no", "source": "sigmorphon_st", "type": "canonical"}
        new_row_unsegmented['ID'] = f"{row_id}_{glottocode}_{i}"
        new_row_unsegmented['transcription'] = line.transcription
        new_row_unsegmented['glosses'] = line.glosses
        new_row_unsegmented['translation'] = line.translation
        data.append(new_row_unsegmented)

    return datasets.Dataset.from_list(data)