import collections
import logging
from typing import Any

import editdistance
import glossing
import pandas as pd
import pytest
import regex as re

from data.model import boundary_pattern

from ..util.type_utils import all_not_none
from .alignment_score import alignment_score

logger = logging.getLogger(__name__)


def evaluate(predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate predictions using appropriate metrics for glossing/segmentation.

    - For gloss predictions, compute metrics such as BLEU and morpheme accuracy.
    - For segmentation predictions, compute metrics such as F1 score.

    If multiple languages are present, we report both overall metrics and metrics per language
    """
    assert {"glottocode", "predicted", "reference", "task"}.issubset(
        predictions.columns
    )
    metrics = {}
    for glottocode, df in predictions.groupby("glottocode"):
        print(f"Evaluating: {glottocode} with {len(df)} rows")
        metrics[glottocode] = _evaluate(df)
    metrics["all"] = _evaluate(predictions)
    return metrics


def _evaluate(predictions: pd.DataFrame):
    gloss_predictions = predictions[predictions["task"].isin(["s2g", "t2g"])]
    segmentation_predictions = predictions[predictions["task"] == "t2s"]

    # Eval for joint t2sg task!!
    if (predictions["task"] == "t2sg").any():
        assert len(gloss_predictions) == 0
        assert len(segmentation_predictions) == 0
        gloss_label = "\nGlosses: "  # Split on this label
        joint_preds = predictions[predictions["task"] == "t2sg"]
        ref_split = (
            joint_preds["reference"]
            .str.split(  # type:ignore
                gloss_label, n=1, expand=True, regex=False
            )
            .fillna("")
        )
        pred_split = (
            joint_preds["predicted"]
            .str.split(  # type:ignore
                gloss_label, n=1, expand=True, regex=False
            )
            .fillna("")
        )
        segmentation_predictions = joint_preds.copy()
        gloss_predictions = joint_preds.copy()
        segmentation_predictions["reference"] = ref_split[0]
        segmentation_predictions["predicted"] = pred_split[0]
        gloss_predictions["reference"] = ref_split[1]
        gloss_predictions["predicted"] = pred_split[1]
    elif (predictions["task"] == "t2sg_interleaved").any():
        assert len(gloss_predictions) == 0
        assert len(segmentation_predictions) == 0
        # TODO: Use `split_interleaved_segments` to split

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
            f"Must have same number of glossing and segmentation predictions. Got {len(gloss_predictions)} glosses and {len(segmentation_predictions)} segmentations."
        )
        joint_predictions = gloss_predictions.merge(
            segmentation_predictions, on="id", suffixes=("_glosses", "_segmentations")
        )
        metrics["alignment"] = alignment_score(
            [
                (s, g)
                for s, g in zip(
                    joint_predictions["predicted_segmentations"].tolist(),
                    joint_predictions["predicted_glosses"].tolist(),
                )
            ]
        )

    return metrics


def _evaluate_segmentation_example(generation: str, label: str):
    assert label is not None

    predicted_words = generation.split()
    label_words = label.split()

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
