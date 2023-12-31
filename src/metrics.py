import numpy as np
import rouge
from pprint import pprint
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    accuracy_score,
    precision_score,
    f1_score,
    cohen_kappa_score,
)


def calc_rouge(txt1: str, txt2: str, stats=None, verbose: bool = True):
    """
    Calculates ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores between two input texts.

    Args:
    - txt1 (str): the first text for comparison
    - txt2 (str): the second text for comparison
    - stats (list): a list of metrics to calculate. It can be one or more of these: ['f', 'p', 'r'] (default: ['f'])
    - verbose (bool): if True, the function prints out the ROUGE scores (default: True)

    Returns:
    - scores (dict): a dictionary containing the ROUGE scores
    """
    if stats is None:
        stats = list("f")
    if stats:
        r = rouge.Rouge(stats=stats)
    else:
        r = rouge.Rouge()
    try:
        scores = r.get_scores(txt1, txt2)[0]
    except RecursionError as e:
        print(f"RecursionError: {e}")
        print("Lengths are:", len(txt1), len(txt2))
        print(f"txt1: {txt1}")
        print(f"txt2: {txt2}")
        scores = {
            "rouge-1": {"f": np.nan},
            "rouge-2": {"f": np.nan},
            "rouge-l": {"f": np.nan},
        }
    if verbose:
        pprint(scores)
    return scores


def calc_rouge_specific(txt1: str, txt2: str, metric: str = "rouge-l", stat="f"):
    """
        Necessary Rouge metric for sentence extraction as specified in https://aclanthology.org/2021.fnp-1.19.
        Joint abstractive and extractive method for long financial document summarization (Zmandar et al, 2021)
    :param txt1:
    :param txt2:
    :param metric:
    :param stat:
    :return:
    """
    return calc_rouge(txt1=txt1, txt2=txt2, stats=[stat], verbose=False)[metric][stat]


def calc_rouge_agg(summary: str, summaries_gold, stats=None, verbose: bool = True):
    if stats is None:
        stats = ["f"]
    assert (
        len(stats) == 1
    ), "Cannot generate a table if there are more than 1 Rouge statistics"
    scores_df = pd.DataFrame()
    for idx, summary_gold in summaries_gold.items():
        tmp_scores = calc_rouge(
            txt1=summary, txt2=summary_gold, stats=stats, verbose=False
        )
        scores_per_gold_summary_df = pd.DataFrame().from_dict(
            tmp_scores, orient="index"
        )
        scores_df = pd.concat([scores_df, scores_per_gold_summary_df], axis=1)
    scores_df.columns = list(summaries_gold.keys())
    scores_df.index.names = ["gold_summary"]
    scores_df = scores_df.transpose()
    if verbose:
        pprint(scores_df)
    return scores_df.mean()


def binary_classification_metrics(true_labels, pred_labels, prefix=""):
    # Find False Positives and True Negatives
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    # Calculate the false positive rate (FPR)
    fpr = fp / (fp + tn)
    # False Negative Rate
    fnr = fn / (tp + fn)
    # Calculate the true negative rate (TNR) - Specificity
    tnr = tn / (tn + fp)
    # Calculate the recall
    recall = recall_score(true_labels, pred_labels)
    # Accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    # Precision
    precision = precision_score(true_labels, pred_labels)
    # F1 Score
    f1_score_ = f1_score(true_labels, pred_labels)
    # Cohen Kappa Score
    cohen_kappa_score_ = cohen_kappa_score(true_labels, pred_labels)

    d = {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1 score": f1_score_,
        "cohen kappa score": cohen_kappa_score_,
        "false positive rate": fpr,
        "true negative rate": tnr,
        "false negative rate": fnr,
    }
    d = {prefix + k: v for k, v in d.items()}
    return d
