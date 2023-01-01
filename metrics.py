import rouge
from pprint import pprint
from thesis.query import get_all_summaries
import pandas as pd


def calc_rouge(txt1: str, txt2: str, stats=None, verbose: bool = True):
    if stats is None:
        stats = list('f')
    if stats:
        r = rouge.Rouge(stats=stats)
    else:
        r = rouge.Rouge()
    scores = r.get_scores(txt1, txt2)[0]
    if verbose:
        pprint(scores)
    return scores


def calc_rouge_agg(summary: str, file_id, training: bool = True, stats=None, verbose: bool = True):
    if stats is None:
        stats = ['f']
    assert len(stats) == 1, 'Cannot generate a table if there are more than 1 Rouge statistics'
    summaries_gold = get_all_summaries(file_id, training)
    scores_df = pd.DataFrame()
    for idx, summary_gold in summaries_gold.items():
        tmp_scores = calc_rouge(txt1=summary, txt2=summary_gold, stats=stats, verbose=False)
        scores_per_gold_summary_df = pd.DataFrame().from_dict(tmp_scores, orient='index')
        scores_df = pd.concat([scores_df, scores_per_gold_summary_df], axis=1)
    scores_df.columns = list(summaries_gold.keys())
    scores_df.index.names = ['gold_summary']
    scores_df = scores_df.transpose()
    if verbose:
        pprint(scores_df)
    return scores_df.mean()
