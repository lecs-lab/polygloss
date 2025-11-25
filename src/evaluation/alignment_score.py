import logging
import random
import re

import editdistance
from glossing.igt import gloss_string_to_word_glosses

logger = logging.getLogger(__name__)


def alignment_score(segments_and_glosses: list[tuple[str, str]]):
    """Computes an alignment score between glossing predictions and segmentation predictions"""
    sum_edit_dist = 0
    # Log a few random examples
    log_indices = random.sample(
        range(len(segments_and_glosses)), k=min(10, len(segments_and_glosses))
    )
    for index, (segments, glosses) in enumerate(segments_and_glosses):
        segments = " ".join(gloss_string_to_word_glosses(segments))
        glosses = " ".join(gloss_string_to_word_glosses(glosses))
        segment_abstract = re.sub(r"[^\=\-\s]+", "x", segments)
        glosses_abstract = re.sub(r"[^\=\-\s]+", "x", glosses)
        edit_distance = editdistance.eval(segment_abstract, glosses_abstract)
        max_edit_distance = max(len(segment_abstract), len(glosses_abstract))
        sentence_score = 1 - (edit_distance / max_edit_distance)

        if index in log_indices:
            logger.info(
                f"Ex {index}\nSegmentation: {segments} ({segment_abstract})\nGlosses: {glosses} ({glosses_abstract})\nScore: {sentence_score}"
            )

        sum_edit_dist += sentence_score
    return sum_edit_dist / len(segments_and_glosses)
