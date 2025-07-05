import collections
import logging
import re

import editdistance
import glossing
import pytest

from src.generate import PredictedExample
from src.util.type_utils import all_not_none

logger = logging.getLogger(__name__)

# TODO: Make sure there aren't any other weird boundaries in our data
DEFAULT_MORPHEME_BOUNDARIES = ["-", "="]


def evaluate(predictions: list[PredictedExample]):
    """Evaluate predictions using appropriate metrics depending on the `output_key`.

    - For gloss predictions, compute metrics such as BLEU and morpheme accuracy.
    - For segmentation predictions, compute metrics such as F1 score.
    """
    gloss_predictions = [p for p in predictions if p.output_key == "glosses"]
    segmentation_predictions = [
        p for p in predictions if p.output_key == "segmentation"
    ]

    metrics: dict[str, dict] = {}

    if len(gloss_predictions) > 0:
        generations = [p.generation for p in gloss_predictions]
        references = [p.label for p in gloss_predictions]
        assert all_not_none(references)
        metrics["glossing"] = glossing.evaluate_glosses(generations, references)

    if len(segmentation_predictions) > 0:
        # Average metrics over examples
        metrics["segmentation"] = collections.defaultdict(float)
        for example_metrics in map(
            _evaluate_segmentation_example, segmentation_predictions
        ):
            for k, v in example_metrics.items():
                metrics["segmentation"][k] += v
        for k in metrics["segmentation"]:
            metrics["segmentation"][k] /= len(segmentation_predictions)

    return metrics


def _evaluate_segmentation_example(example: PredictedExample):
    assert example.label is not None
    assert example.output_key == "segmentation"

    predicted_words = example.generation.split()
    label_words = example.label.split()

    boundary_pattern = re.compile("|".join(DEFAULT_MORPHEME_BOUNDARIES))
    predicted_morphemes = [re.split(boundary_pattern, word) for word in predicted_words]
    label_morphemes = [re.split(boundary_pattern, word) for word in label_words]

    # Compute modified f1
    total_predicted_morphs = sum(len(w) for w in predicted_morphemes)
    total_label_morphs = sum(len(w) for w in label_morphemes)
    total_overlapping = sum(
        _intersect_size(pred, label)
        for pred, label in zip(predicted_morphemes, label_morphemes)
    )
    precision = total_overlapping / total_predicted_morphs
    recall = total_overlapping / total_label_morphs
    f1 = (2 * precision * recall) / (precision + recall)

    edit_dist = editdistance.eval(" ".join(predicted_words), " ".join(label_words))

    return {
        "accuracy": int(predicted_words == label_words),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "edit_distance": edit_dist,
    }


def _intersect_size(l1: list[str], l2: list[str]):
    """Computes the size of the intersection, allowing duplicate items, of two lists.
    i.e., if one list has the same element twice and the other has it once, we count this as 1
    """
    c1 = collections.Counter(l1)
    c2 = collections.Counter(l2)
    return sum(min(c1[k], c2[k]) for k in set(l1).intersection(set(l2)))


def test_evaluate_segmentation_example():
    example = PredictedExample(
        generation="t-he cat-s are run-ing",
        label="the cat-s are runn-ing",
        output_key="segmentation",
    )
    metrics = _evaluate_segmentation_example(example)
    assert metrics["accuracy"] == 0
    assert metrics["precision"] == 4 / 7
    assert metrics["recall"] == 4 / 6
    assert metrics["f1"] == pytest.approx(2 / ((7 / 4) + (6 / 4)))
