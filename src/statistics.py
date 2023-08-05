import os
from pprint import pprint

from tqdm import tqdm

from preprocessing import clean, preprocess
from query import get_raw_data_dir, get_report, get_all_summaries, get_file_handles
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd

from metrics import calc_rouge_agg


def compute_corpus_stats(raw_data: bool = True, training: bool = True):
    if not raw_data:
        return {}
    data_dir = get_raw_data_dir(training=training)
    corpus = PlaintextCorpusReader(data_dir, r".*\.txt")
    d = {
        "raw_data": raw_data,
        "training": training,
        "chars": sum([len(w) for w in corpus.words()]),
        "words": len(corpus.words()),
        "sents": len(corpus.sents()),
    }
    return d


def compute_statistics_on_document(doc: str):
    doc_str, doc_obj = preprocess(doc)
    words = word_tokenize(doc_str)
    sents = sent_tokenize(doc_str)
    d = {
        "chars": sum([len(w) for w in words]),
        "words": len(words),
        "sents": len(sents),
    }
    return d


def calc_rouge_agg_from_gold_summaries(
    summary: str, file_id, training: bool = True, stats=None, verbose: bool = True
):
    summaries_gold = get_all_summaries(file_id, training)
    return calc_rouge_agg(
        summary=summary, summaries_gold=summaries_gold, stats=stats, verbose=verbose
    )


def get_stats_gold_summaries_extraction(training: bool = True, save_file: bool = True):
    """
        Check if after preprocessing both annual reports and its gold summaries, the latter exist \
        in the respective report. If this is the case for most of the reports, there will not be a \
        need for complex sentence matching.
    """
    extraction_stats = []
    for file_path in tqdm(get_file_handles(training=training, gold=False)):
        file_id = file_path.split("/")[-1]
        report = get_report(file_id=file_id, training=training)
        report, _ = preprocess(report)
        summaries = get_all_summaries(file_id=file_id, training=training)
        nonexistent_summaries = 0
        existent_summaries = 0
        problem_files = ""
        for summary_id, s in summaries.items():
            summary, _ = preprocess(s)
            if summary not in report:
                nonexistent_summaries += 1
                problem_files += str(summary_id) + ","
                print("From report:", file_id)
                print(
                    "Summary:",
                    summary_id,
                    "has missing sentences (problems with extraction)",
                )
                print(summary)
            else:
                existent_summaries += 1

        extraction_stats_per_report = {
            "report": file_id,
            "existent_summaries": existent_summaries,
            "nonexistent_summaries": nonexistent_summaries,
            "nonexistent_summaries_files": problem_files,
        }
        extraction_stats.append(extraction_stats_per_report)
    extraction_stats_df = pd.DataFrame(extraction_stats)
    if save_file:
        os.makedirs("../tmp", exist_ok=True)
        data_type = "training" if training else "validation"
        extraction_stats_df.to_csv(f"tmp/gold_summaries_extraction_{data_type}.csv")
    return extraction_stats_df


def main():
    report = get_report(17)
    report_clean = clean(report)
    stats = compute_statistics_on_document(report)
    stats_clean = compute_statistics_on_document(report_clean)
    pprint(stats)
    pprint(stats_clean)


# main()
