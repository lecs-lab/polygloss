import re

import editdistance


def alignment_score(segments_and_glosses: list[tuple[str, str]]):
    """Computes an alignment score between glossing predictions and segmentation predictions"""
    sum_edit_dist = 0
    for segments, glosses in segments_and_glosses:
        segment_abstract = re.sub(r"[^\=\-\s]+", "x", segments)
        glosses_abstract = re.sub(r"[^\=\-\s]+", "x", glosses)
        sum_edit_dist += editdistance.eval(segment_abstract, glosses_abstract)
    return -1 * sum_edit_dist / len(segments_and_glosses)
