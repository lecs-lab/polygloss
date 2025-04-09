import regex
from langid.langid import LanguageIdentifier, model

from data.lang_codes import iso1_to_3, iso3_to_glotto
from data.model import IGTLine

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
identifier.set_languages(["zh", "nl", "en", "es", "pt", "de", "fr", "it"])


def guess_metalang(example: IGTLine, confidence_threshold=0.95) -> IGTLine:
    """If the example is missing a metalanguage code, try to predict one with langid"""
    if example.metalang_glottocode is not None or example.translation is None:
        return example
    if isinstance(example.translation, float):
        breakpoint()
    iso1code, confidence = identifier.classify(example.translation)
    if confidence >= confidence_threshold:
        iso3code = iso1_to_3(iso1code)
        glottocode = iso3_to_glotto.get(iso3code)
        example.metalang_glottocode = glottocode
    return example


def standardize(example: IGTLine) -> IGTLine:
    if not isinstance(example.translation, str):
        example.translation = None

    # Fix punctuation and such
    example.transcription = regex.sub(r"(\w)\?", r"\1 ?", example.transcription)
    example.transcription = regex.sub(r"(\w)\.", r"\1 .", example.transcription)
    example.transcription = regex.sub(r"(\w)\!", r"\1 !", example.transcription)
    example.transcription = regex.sub(r"(\w)\,", r"\1 ,", example.transcription)
    example.transcription = regex.sub(r"\-(\s|$)", " ", example.transcription)

    # I've omitted a regex that removes null morphemes (e.g. dangling hyphens)
    # If you want to add that back, the regex is:
    # regex.sub("\-(\s|$)", " ")

    if example.glosses:
        example.glosses = regex.sub("\t", " ", example.glosses)
        example.glosses = regex.sub(r"(\w)\.(\s|$)", r"\1 . ", example.glosses)
        example.glosses = regex.sub(r"(\w)\!(\s|$)", r"\1 ! ", example.glosses)
        example.glosses = regex.sub(r"(\w)\?(\s|$)", r"\1 ? ", example.glosses)

    return example
