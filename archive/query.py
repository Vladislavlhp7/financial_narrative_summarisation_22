import os
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.metrics import calc_rouge_specific
from src.preprocessing import tokenized_sent_to_str, preprocess
from src.query import get_all_summaries, get_report, get_file_handles_binary, get_binary_labels_data_dir


def get_most_similar_sentence(gold_summary_sent: str, report_preprocessed) -> Tuple[int, str]:
    assert len(gold_summary_sent) > 0, 'Gold summary has an invalid length of 0'
    # Most times the sentences are already in the gold summary (as they are extracted)
    for i, sent_tokenized in enumerate(report_preprocessed.sentences):
        sent = tokenized_sent_to_str(sent_tokenized)
        if sent == gold_summary_sent:  # speed up the process and skip checking
            return i, sent

    # When the same sentence is not existent (highly unlikely) try to maximise from Rouge
    sent_rouge_scores = []
    for i, sent_tokenized in enumerate(report_preprocessed.sentences):
        sent = tokenized_sent_to_str(sent_tokenized)
        rouge_score = calc_rouge_specific(sent, gold_summary_sent)
        sent_rouge_scores.append(rouge_score)
    best_report_sent_idx = int(np.argmax(sent_rouge_scores))
    best_report_sent = tokenized_sent_to_str(report_preprocessed.sentences[best_report_sent_idx])
    return best_report_sent_idx, best_report_sent


def get_most_similar_sentences(gold_summaries_preprocessed_dict, report_preprocessed) -> Tuple[
    Dict[str, List[str]], List[int]]:
    """
        Match each sentence from the gold summary with the one from the report which maximises the Rouge metric as specified in:
        - Chen et al., 2018. Fast abstractive summarization with reinforce-selected sentence rewriting

        - Nallapati et al., 2016. Abstractive text summarization using sequence-to-sequence RNNs and beyond

        - Zmandar et al., 2021. Joint abstractive and extractive method for long financial document summarization
    :param gold_summaries_preprocessed_dict: cleaned, preprocessed and tokenized summary
    :param report_preprocessed: cleaned, preprocessed and tokenized report
    :return: mapping between gold sentence and report sentence, and sentence indices in annual report
    """
    sim_sent_dict = {}
    maximising_report_sent_ind_arr = []
    for i, sent_tokenized in enumerate(gold_summaries_preprocessed_dict.sentences):
        gold_sent = tokenized_sent_to_str(sent_tokenized)
        best_report_sent_idx, best_report_sent = get_most_similar_sentence(gold_summary_sent=gold_sent,
                                                                           report_preprocessed=report_preprocessed)
        sim_sent_dict[gold_sent] = report_preprocessed.sentences[best_report_sent_idx]
        maximising_report_sent_ind_arr.append(best_report_sent_idx)
    return sim_sent_dict, maximising_report_sent_ind_arr


def match_maximizing_summary_with_report(gold_summaries_preprocessed_dict, report_preprocessed):
    """
        "For each summary sentence exactly one document sentence is matched, [...]
Eventually summary level ROUGE scores are calculated and summary with maximum
score is chosen for further processing and training" (Zmandar et al., 2021)
    :param report_preprocessed:
    :param gold_summaries_preprocessed_dict:
    :return:
    """
    # Get greedy summary-level sentence-matching between summary and report
    sent_matching_dict = {}
    # keep track of indices of report sentences on a gold summary level
    maximising_report_sent_ind_arr_per_summary = {}
    maximising_report_sent_ind_arr = []
    for gold_summary_filename, gold_summary_split in gold_summaries_preprocessed_dict.items():
        sent_matching_per_gold_summary, maximising_report_sent_ind_arr = get_most_similar_sentences(
            gold_summaries_preprocessed_dict=gold_summary_split,
            report_preprocessed=report_preprocessed)
        maximising_report_sent_ind_arr_per_summary[gold_summary_filename] = maximising_report_sent_ind_arr
        sent_matching_dict[gold_summary_filename] = sent_matching_per_gold_summary

    # Select Rouge-maximizing summary
    max_score = 0
    maximising_gold_summary = ""
    maximising_report_match = ""
    for gold_summary_filename, sent_matching_per_gold_summary in sent_matching_dict.items():
        gold_summary = " ".join(sent_matching_per_gold_summary.keys())
        # Generate string from sentence objects
        matched_report_sentences = sent_matching_per_gold_summary.values()
        matched_report = ""
        for report_sentence in matched_report_sentences:
            matched_report += tokenized_sent_to_str(report_sentence) + ' '
        rouge_score = calc_rouge_specific(matched_report, gold_summary)
        if rouge_score > max_score:
            max_score = rouge_score
            maximising_report_match = matched_report
            maximising_gold_summary = gold_summary
            maximising_report_sent_ind_arr = maximising_report_sent_ind_arr_per_summary[gold_summary_filename]
    return maximising_report_match, maximising_gold_summary, maximising_report_sent_ind_arr


def match_maximizing_summary_with_report_by_file_id(file_id, training: bool = True):
    """
        "For each summary sentence exactly one document sentence is matched, [...]
Eventually summary level ROUGE scores are calculated and summary with maximum
score is chosen for further processing and training" - (Zmandar et al., 2021)
    :param file_id:
    :param training:
    :return:
    """
    report = get_report(file_id=file_id)  # [:5_000]  # for test mode
    gold_summaries = get_all_summaries(file_id=file_id, training=training)

    # Preprocess documents
    _, report_preprocessed = preprocess(report)
    summaries_preprocessed_dict = {}
    for file_id_str, gold_summary in gold_summaries.items():
        _, summaries_preprocessed_dict[file_id_str] = preprocess(gold_summary)
    maximising_report_match, maximising_gold_summary, _ = match_maximizing_summary_with_report(
        gold_summaries_preprocessed_dict=summaries_preprocessed_dict,
        report_preprocessed=report_preprocessed)
    return maximising_report_match, maximising_gold_summary


def get_report_sentences_binary_labels_by_rouge_maximisation(file_id, training: bool = True):
    """
            Provide a mapping between each sentence and its binary label of whether \
            it is part of the maximal gold summary
        :param file_id:
        :param training:
        :return:
        """
    report = get_report(file_id=file_id, training=training)  # [:5_000]  # for test mode
    gold_summaries = get_all_summaries(file_id=file_id, training=training)

    # Preprocess documents
    _, report_preprocessed = preprocess(report)
    summaries_preprocessed_dict = {}
    for idx, gold_summary in gold_summaries.items():
        _, summaries_preprocessed_dict[idx] = preprocess(gold_summary)
    _, _, maximising_report_sent_ind_arr = match_maximizing_summary_with_report(
        gold_summaries_preprocessed_dict=summaries_preprocessed_dict,
        report_preprocessed=report_preprocessed)
    # Create the mapping between annual report sentences and their binary labels regarding maximal gold summary
    sent_label_mapping = {}
    for i, sentence in enumerate(report_preprocessed.sentences):
        sent_label_mapping[tokenized_sent_to_str(sentence)] = int(i in maximising_report_sent_ind_arr)
    return sent_label_mapping


def get_sentence_embedding_data_per_file(model, file_id, training: bool = True, store_df: bool = True):
    path = get_binary_labels_data_dir(training=training) + file_id + '.csv'
    df = pd.read_csv(path)
    # df['file_id'] = int(file_id)
    X_sent = df['sent']
    X = np.array([model.wv.get_sentence_vector(sent) for sent in X_sent])
    df['sent_embedding'] = X.tolist()
    # assert (df.columns == ['sent', 'label', 'sent_embedding']).all()
    df = df[['sent', 'label', 'sent_embedding']]
    if store_df:
        t = datetime.now().strftime("%Y-%m-%d %H")
        data_type = 'training' if training else 'validation'
        f_dir = '../tmp/sent_embed'
        os.makedirs(f_dir, exist_ok=True)
        f = f'{f_dir}/{data_type}_{file_id}_{t}.csv'
        df = df.reset_index(drop=True)
        df.to_csv(f)
    return df


def get_sentence_embedding_data_from_corpus(model, training: bool = True, store_df: bool = True):
    """
        Return a ready-to-use embedding and label arrays.
    """
    files = get_file_handles_binary()
    dfs = []
    data_type = 'training' if training else 'validation'
    for file_id in tqdm(files.keys(), 'Generating sentence-level embeddings from corpus'):
        df = get_sentence_embedding_data_per_file(model=model, file_id=file_id, training=training)
        dfs.append(df)
    dfs = pd.concat(dfs)
    X = dfs['sent_embedding']
    y = np.array(dfs['label']).reshape(-1, 1)
    if store_df:
        t = datetime.now().strftime("%Y-%m-%d %H-%M")
        f = f'tmp/{data_type}_corpus_{t}.csv'
        print(f'Storing corpus embedding df at {f}')
        dfs = dfs.reset_index(drop=True)
        dfs.to_csv(f)
    return X, y
