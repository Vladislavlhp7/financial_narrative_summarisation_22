import os
import random
from typing import Dict, Union

import pandas as pd
from tqdm import tqdm

from preprocessing import clean_company_name, preprocess


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


def generate_binary_labels_for_data(training: bool = True, gold: bool = False):
    """
    Create a csv file containing a mapping between preprocessed sentences and a binary score which represents \
    whether the sentence is found in the gold summaries. Compute simple statistics like word count
    """
    for file_path in tqdm(get_file_handles(training=training, gold=gold)):
        file_id = file_path.split('/')[-1]
        try:
            binary_file_path = get_binary_labels_data_dir(training=training, gold=gold) + file_id + '.csv'
            if not os.path.exists(binary_file_path):
                sent_label_mapping = get_report_sentences_binary_labels_from_str(file_id=file_id, training=training,
                                                                                 all_summaries=True)
                sent_label_mapping_df = pd.DataFrame().from_dict(sent_label_mapping, orient='index').reset_index()
                sent_label_mapping_df.columns = ['sent', 'label']
                # Compute simple statistics
                sent_label_mapping_df['words_count'] = sent_label_mapping_df['sent'].apply(lambda x: len(x.split()) + 1)
                sent_label_mapping_df.to_csv(binary_file_path, index=False)
        except Exception as e:
            print('------------------------------------------')
            print(f'{str(e)} at file {file_id}')
            print('------------------------------------------')


def main():
    generate_binary_labels_for_data(training=True)
    print(os.getcwd())
    pass


main()
