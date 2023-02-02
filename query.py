import os
import random
from typing import Dict, Union, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from metrics import calc_rouge_agg, calc_rouge_specific
from preprocessing import clean_company_name, preprocess, tokenized_sent_to_str


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


def get_raw_data_dir(training: bool = True) -> str:
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    path += 'annual_reports/'
    return path


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


def get_report_sentences_binary_labels_from_str(file_id, training: bool = True, all_summaries: bool = True,
                                                summary_id=None):
    """
         Provide a mapping between each sentence and its binary label of whether \
            it is part of all/any summary/ies
    """
    report = get_report(file_id=file_id, training=training)
    if not all_summaries and summary_id is not None:
        gold_summaries = get_summary(file_id=file_id, summary_id=summary_id)
    else:
        gold_summaries = get_all_summaries(file_id=file_id, training=training)
    # Preprocess report as list of sentences
    _, report_preprocessed = preprocess(report)
    summaries_preprocessed_dict = {}
    # Process summaries as whole strings
    for idx, gold_summary in gold_summaries.items():
        summaries_preprocessed_dict[idx], _ = preprocess(gold_summary)
    # Create the mapping between annual report sentences and their binary labels
    sent_label_mapping = {}
    # Iterate over all preprocessed report sentences and if they exist in any summary record 1, otherwise 0
    for i, sentence in enumerate(report_preprocessed.sentences):
        sentence = sentence.strip()
        sent_found = False
        for summary in summaries_preprocessed_dict.values():
            if sentence in summary:
                sent_label_mapping[sentence] = 1
                sent_found = True
                break
        if not sent_found:
            sent_label_mapping[sentence] = 0
    return sent_label_mapping


def get_binary_labels_data_dir(training: bool = True, gold: bool = False):
    file_path = 'data/'
    if training:
        file_path += 'training_binary/'
    else:
        file_path += 'validation_binary/'
    if gold:
        file_path += 'gold_summaries/'
    else:
        file_path += 'annual_reports/'
    os.makedirs(file_path, exist_ok=True)
    return file_path


def get_file_handles_binary(training: bool = True) -> Dict[str, str]:
    """
        Retrieve file handles for training and validation report sentences and binary labels
    :param training: bool
    :return: dict of file paths
    """
    path = get_binary_labels_data_dir(training=training)
    if not os.path.exists(path):
        print(f'Specified path: {path} does not exist')
        return {}
    file_handles = {}
    for f in os.listdir(path):
        report_id = f.replace('.csv', '')
        file_handles[report_id] = f'{path}{f}'
    return file_handles


def generate_binary_labels_for_data(training: bool = True, gold: bool = False, rouge_maximisation: bool = False):
    for file_path in tqdm(get_file_handles(training=training, gold=gold)):
        file_id = file_path.split('/')[-1]
        try:
            binary_file_path = get_binary_labels_data_dir(training=training, gold=gold) + file_id + '.csv'
            if not os.path.exists(binary_file_path):
                if not rouge_maximisation:
                    sent_label_mapping = get_report_sentences_binary_labels_from_str(file_id=file_id, training=training,
                                                                                     all_summaries=True)
                else:
                    sent_label_mapping = get_report_sentences_binary_labels_by_rouge_maximisation(file_id=file_id,
                                                                                                  training=training)
                sent_label_mapping_df = pd.DataFrame().from_dict(sent_label_mapping, orient='index').reset_index()
                sent_label_mapping_df.columns = ['sent', 'label']
                # Compute simple statistics
                sent_label_mapping_df['words_count'] = sent_label_mapping_df['sent'].apply(lambda x: len(x.split()) + 1)
                sent_label_mapping_df.to_csv(binary_file_path, index=False)
        except Exception as e:
            print('------------------------------------------')
            print(f'{str(e)} at file {file_id}')
            print('------------------------------------------')


def calc_rouge_agg_from_gold_summaries(summary: str, file_id, training: bool = True, stats=None, verbose: bool = True):
    summaries_gold = get_all_summaries(file_id, training)
    return calc_rouge_agg(summary=summary, summaries_gold=summaries_gold, stats=stats, verbose=verbose)


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
        f_dir = 'tmp/sent_embed'
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


def main():
    generate_binary_labels_for_data(training=False)
    pass


main()
