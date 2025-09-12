import collections
import logging
import re
from typing import Any

import editdistance
import glossing
import pandas as pd
import pytest

from alignment_score import alignment_score
from src.util.type_utils import all_not_none

logger = logging.getLogger(__name__)

# TODO: Make sure there aren't any other weird boundaries in our data
DEFAULT_MORPHEME_BOUNDARIES = ["-", "="]


def evaluate(predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate predictions using appropriate metrics for glossing/segmentation.

    - For gloss predictions, compute metrics such as BLEU and morpheme accuracy.
    - For segmentation predictions, compute metrics such as F1 score.

    If multiple languages are present, we report both overall metrics and metrics per language
    """
    assert {"glottocode", "predicted", "reference", "input_key", "output_key"}.issubset(
        predictions.columns
    )
    metrics = {}
    for glottocode, df in predictions.groupby("glottocode"):
        metrics[glottocode] = _evaluate(df)
    metrics["all"] = _evaluate(predictions)
    return metrics


def _evaluate(predictions: pd.DataFrame):
    gloss_predictions = predictions[predictions["output_key"] == "glosses"]
    segmentation_predictions = predictions[predictions["output_key"] == "segmentation"]

    metrics: dict[str, dict | float] = {}

    if len(gloss_predictions) > 0:
        generations = gloss_predictions["predicted"].tolist()
        references = gloss_predictions["reference"].tolist()
        assert all_not_none(references)
        metrics["glossing"] = glossing.evaluate_glosses(generations, references)

    if len(segmentation_predictions) > 0:
        # Average metrics over examples

        segmentation_metrics = collections.defaultdict(float)
        for _, row in segmentation_predictions.iterrows():
            for k, v in _evaluate_segmentation_example(
                row["predicted"],  # type:ignore
                row["reference"],  # type:ignore
            ).items():
                segmentation_metrics[k] += v
        segmentation_metrics = {
            k: v / len(segmentation_predictions)
            for k, v in segmentation_metrics.items()
        }
        metrics["segmentation"] = segmentation_metrics

    if len(gloss_predictions) > 0 and len(segmentation_predictions) > 0:
        # We have both, let's calculate alignment
        assert len(gloss_predictions) == len(segmentation_predictions), (
            "Must have same number of glossing and segmentation predictions."
        )
        joint_predictions = gloss_predictions.join(
            segmentation_predictions,
            on="id",
            lsuffix="glossing_",
            rsuffix="segmentation_",
        )
        metrics["alignment"] = alignment_score(
            [
                (g, s)
                for g, s in zip(
                    joint_predictions["glossing_predicted"].tolist(),
                    joint_predictions["segmentation_predicted"].tolist(),
                )
            ]
        )

    return metrics


def _evaluate_segmentation_example(generation: str, label: str):
    assert label is not None

    predicted_words = generation.split()
    label_words = label.split()

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
    precision = (
        total_overlapping / total_predicted_morphs if total_predicted_morphs > 0 else 0
    )
    recall = total_overlapping / total_label_morphs
    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

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
    metrics = _evaluate_segmentation_example(
        generation="t-he cat-s are run-ing",
        label="the cat-s are runn-ing",
    )
    assert metrics["accuracy"] == 0
    assert metrics["precision"] == 4 / 7
    assert metrics["recall"] == 4 / 6
    assert metrics["f1"] == pytest.approx(2 / ((7 / 4) + (6 / 4)))
