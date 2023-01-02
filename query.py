import os
import random
from typing import Dict, Union, List, Tuple

import numpy as np

from thesis.metrics import calc_rouge_agg, calc_rouge_specific
from thesis.preprocessing import clean_company_name, preprocess, tokenized_sent_to_str


def get_file_handles(training: bool = True, gold: bool = False) -> Dict[str, str]:
    """
        Retrieve file handles for training and validation reports and gold summaries
    :param training: bool
    :param gold: bool
    :return: dict of file paths
    """
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    data_type = 'gold_summaries/' if gold else 'annual_reports/'
    path += data_type

    if not os.path.exists(path):
        print(f'Specified path: {path} does not exist')
        return {}
    file_handles = {}
    for f in os.listdir(path):
        report_id = f.replace('.txt', '')
        file_handles[report_id] = f'{path}{f}'
    return file_handles


def get_gold_summaries_file_handles(file_id, training: bool = True) -> Dict[str, str]:
    """
        There are a few gold summaries per report, and they are enumerated as <report>_<#summary>.txt
    :param file_id: str/int
    :param training: bool
    :return: list of file paths
    """
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    path += 'gold_summaries/'
    if not os.path.exists(path):
        print(f'Specified path: {path} does not exist')
        return {}

    file_handles = {}
    for f in os.listdir(path):
        report_id = f.split('_')[0]
        summary_id = f.split('_')[1].replace('.txt', '')
        if str(report_id) == str(file_id):
            file_handles[str(summary_id)] = f'{path}{f}'
    return file_handles


def get_company_from_id(file_id, training: bool = True) -> Union[str, None]:
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    path += 'annual_reports/'
    path += str(file_id)
    path += '.txt'
    max_line_iter = 100  # assume company name is before the 100th line to save on reading time
    with open(path) as file:
        for i, line in enumerate(file):
            if ' plc' in line:
                return clean_company_name(line)
            if i > max_line_iter:
                print('Name not found')
                return None


def get_id_to_company_mapping(training: bool = True) -> Dict[str, str]:
    """
        Many-to-one relationship, as reports are issued per year for a company
    :param training:
    :return:
    """
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    path += 'annual_reports/'
    id_to_company_dict = {}
    for f in os.listdir(path):
        report_id = f.replace('.txt', '')
        id_to_company_dict[report_id] = get_company_from_id(report_id)
    return id_to_company_dict


def get_report(file_id, training: bool = True) -> str:
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    path += 'annual_reports/'
    path += str(file_id) + '.txt'
    f = open(path, "r")
    return f.read()


def get_summary(file_id, summary_id=None, training: bool = True) -> str:
    """
        Return a specific (or random) summary for a specific report
    :param file_id:
    :param summary_id:
    :param training:
    :return:
    """
    file_handles = get_gold_summaries_file_handles(file_id, training)
    if summary_id:
        path = file_handles[str(summary_id)]
    else:
        # select random summary
        path = random.choice(list(file_handles.values()))
    with open(path, "r") as f:
        summary = f.read()
    return summary


def get_all_summaries(file_id, training: bool = True) -> Dict[str, str]:
    """
        Return all summaries (str) for a report
    :param file_id:
    :param training:
    :return:
    """
    file_handles = get_gold_summaries_file_handles(file_id, training)
    summaries = {}
    for idx, path in file_handles.items():
        with open(path, "r") as f:
            summary = f.read()
        summaries[str(idx)] = summary
    return summaries


def get_most_similar_sentence(gold_summary_sent: str, report_preprocessed) -> Tuple[int, str]:
    assert len(gold_summary_sent) > 0, 'Gold summary has an invalid length of 0'
    sent_rouge_scores = []
    for i, sent_tokenized in enumerate(report_preprocessed.sentences):
        sent = tokenized_sent_to_str(sent_tokenized)
        rouge_score = calc_rouge_specific(sent, gold_summary_sent)
        sent_rouge_scores.append(rouge_score)
    best_report_sent_idx = int(np.argmax(sent_rouge_scores))
    best_report_sent = tokenized_sent_to_str(report_preprocessed.sentences[best_report_sent_idx])
    return best_report_sent_idx, best_report_sent


def get_most_similar_sentences(gold_summaries_preprocessed_dict, report_preprocessed) -> Dict[str, List[str]]:
    """
        Match each sentence from the gold summary with the one from the report
        which maximises the Rouge metric as specified in:
        - Chen et al., 2018. Fast abstractive summarization with reinforce-selected sentence rewriting
        - Nallapati et al., 2016. Abstractive text summarization using sequence-to-sequence RNNs and beyond
        - Zmandar et al., 2021. Joint abstractive and extractive method for long financial document summarization
    :param gold_summaries_preprocessed_dict: cleaned, preprocessed and tokenized summary
    :param report_preprocessed: cleaned, preprocessed and tokenized report
    :return: mapping between gold sentence and report sentence
    """
    sim_sent_dict = {}
    for i, sent_tokenized in enumerate(gold_summaries_preprocessed_dict.sentences):
        gold_sent = tokenized_sent_to_str(sent_tokenized)
        best_report_sent_idx, best_report_sent = get_most_similar_sentence(gold_summary_sent=gold_sent,
                                                                           report_preprocessed=report_preprocessed)
        sim_sent_dict[gold_sent] = report_preprocessed.sentences[best_report_sent_idx]
    return sim_sent_dict


def match_maximizing_summary_with_report(gold_summaries_preprocessed_dict, report_preprocessed):
    """
        "For each summary sentence exactly one document sentence is matched, [...]
         Eventually summary level ROUGE scores are calculated and summary with maximum
         score is chosen for further processing and training" - (Zmandar et al., 2021)
    :param report_preprocessed:
    :param gold_summaries_preprocessed_dict:
    :return:
    """
    # Get greedy summary-level sentence-matching between summary and report
    sent_matching_dict = {}
    for idx, gold_summary_split in gold_summaries_preprocessed_dict.items():
        sent_matching_per_gold_summary = get_most_similar_sentences(gold_summaries_preprocessed_dict=gold_summary_split,
                                                                    report_preprocessed=report_preprocessed)
        sent_matching_dict[idx] = sent_matching_per_gold_summary

    # Select Rouge-maximizing summary
    max_score = 0
    maximising_gold_summary = ""
    maximising_report_match = ""
    for idx, sent_matching_per_gold_summary in sent_matching_dict.items():
        gold_summary = " ".join(sent_matching_per_gold_summary.keys())
        # Generate string from sentence objects
        matched_report_sentences = sent_matching_per_gold_summary.values()
        matched_report = ""
        for report_sentence in matched_report_sentences:
            matched_report += tokenized_sent_to_str(report_sentence) + ' '
        # Deal with tokenized apostrophes
        matched_report = matched_report.replace(" 's", "'s")
        rouge_score = calc_rouge_specific(matched_report, gold_summary)
        if rouge_score > max_score:
            max_score = rouge_score
            maximising_report_match = matched_report
            maximising_gold_summary = gold_summary
    return maximising_report_match, maximising_gold_summary


def match_maximizing_summary_with_report_by_file_id(file_id, training: bool = True):
    """
        "For each summary sentence exactly one document sentence is matched, [...]
         Eventually summary level ROUGE scores are calculated and summary with maximum
         score is chosen for further processing and training" - (Zmandar et al., 2021)
    :param file_id:
    :param training:
    :return:
    """
    report = get_report(file_id=file_id)[:5_000]
    gold_summaries = get_all_summaries(file_id=file_id, training=training)

    # Preprocess documents
    _, report_preprocessed = preprocess(report)
    summaries_preprocessed_dict = {}
    for idx, gold_summary in gold_summaries.items():
        _, summaries_preprocessed_dict[idx] = preprocess(gold_summary)
    return match_maximizing_summary_with_report(gold_summaries_preprocessed_dict=summaries_preprocessed_dict,
                                                report_preprocessed=report_preprocessed)


def calc_rouge_agg_from_gold_summaries(summary: str, file_id, training: bool = True, stats=None, verbose: bool = True):
    summaries_gold = get_all_summaries(file_id, training)
    return calc_rouge_agg(summary=summary, summaries_gold=summaries_gold, stats=stats, verbose=verbose)


def main():
    # print(get_file_handles()['17'])
    # file_handles = get_gold_summaries_file_handles(17)
    # print(file_handles)
    # print(get_company_from_id('17'))
    # pprint(get_id_to_company_mapping())

    # ################################################
    # report = get_report(17)
    # gold_summary = get_summary(17)
    # _, gold_summary_split = preprocess(gold_summary)
    # _, report_split = preprocess(report)
    # d = get_most_similar_sentences(gold_summary_split=gold_summary_split, report_split=report_split)
    # print(d)

    # ################################################
    maximising_report_match, maximising_gold_summary = match_maximizing_summary_with_report_by_file_id(17)
    print(maximising_report_match)
    print(maximising_gold_summary)


main()
