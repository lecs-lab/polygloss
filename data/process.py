import regex
from langid.langid import LanguageIdentifier, model
from pyglottolog import Glottolog

from data.lang_codes import iso1_to_3, iso3_to_glotto
from data.model import IGTLine

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
identifier.set_languages(["zh", "nl", "en", "es", "pt", "de", "fr", "it"])

try:
    glottolog = Glottolog("../glottolog")
except ValueError:
    raise Exception(
        "You must download Glottolog and put it in the same directory as the polygloss repo. See https://github.com/glottolog/pyglottolog#install"
    )

lang_cache: dict[str, str] = {}


def get_lang(glottocode: str):
    """Convert a glottocode into a language name. Cached."""
    if glottocode in lang_cache:
        return lang_cache[glottocode]
    languoid = glottolog.languoid(glottocode)
    lang_cache[glottocode] = languoid.name if languoid and languoid.name else "Unknown"
    return lang_cache[glottocode]


def add_lang_info(example: IGTLine, confidence_threshold=0.95) -> IGTLine:
    # If the example is missing a metalanguage code, try to predict one with langid
    if example.metalang_glottocode is None and example.translation:
        iso1code, confidence = identifier.classify(example.translation)
        if confidence >= confidence_threshold:
            iso3code = iso1_to_3(iso1code)
            glottocode = iso3_to_glotto.get(iso3code)
            example.metalang_glottocode = glottocode

    # Look up glottocodes to get languages
    if example.glottocode:
        example.language = get_lang(example.glottocode)
    if example.metalang_glottocode:
        example.metalanguage = get_lang(example.metalang_glottocode)

    return example


def standardize(example: IGTLine) -> IGTLine:
    if not isinstance(example.translation, str):
        example.translation = None

    # Fix punctuation and such
    def _fix_punc(s: str):
        s = regex.sub(r"(\w)\?", r"\1 ?", s)
        s = regex.sub(r"(\w)\.", r"\1 .", s)
        s = regex.sub(r"(\w)\!", r"\1 !", s)
        s = regex.sub(r"(\w)\,", r"\1 ,", s)
        return s

    example.transcription = _fix_punc(example.transcription)
    example.segmentation = (
        _fix_punc(example.segmentation) if example.segmentation else None
    )

    # I've omitted a regex that removes null morphemes (e.g. dangling hyphens)
    # If you want to add that back, the regex is:
    # regex.sub("\-(\s|$)", " ")

    if example.glosses:
        example.glosses = regex.sub("\t", " ", example.glosses)
        example.glosses = _fix_punc(example.glosses)

    return example
