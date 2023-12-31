import os
import pickle
import random
from datetime import datetime
from typing import Dict, Union, List

import pandas as pd
from gensim.models import FastText
from nltk import word_tokenize
from tqdm import tqdm

from preprocessing import clean_company_name, preprocess


def get_file_handles(
    training: bool = True, gold: bool = False, root: str = ".."
) -> Dict[str, str]:
    """
    Returns a dictionary of file paths for either the training or validation dataset and either the annual reports or gold summaries.

    Args:
        training (bool): A boolean to indicate whether the training data is being used.
        gold (bool): A boolean to indicate whether the gold summaries are being used.
        root (str): The root directory of the data files.

    Returns:
        Dict[str, str]: A dictionary with keys as the report IDs and values as the file paths for each file in the directory.
    """
    path = f"{root}/data/"
    data_type = "training/" if training else "validation/"
    path += data_type
    data_type = "gold_summaries/" if gold else "annual_reports/"
    path += data_type

    if not os.path.exists(path):
        print(f"Specified path: {path} does not exist")
        return {}
    file_handles = {}
    for f in os.listdir(path):
        report_id = f.replace(".txt", "")
        file_handles[report_id] = f"{path}{f}"
    return file_handles


def get_gold_summaries_file_handles(
    file_id, training: bool = True, root: str = ".."
) -> Dict[str, str]:
    """
    Returns a dictionary of file paths for each summary of a given report.

    Args:
        file_id: The ID of the report to find the summaries for.
        training (bool): A boolean to indicate whether the training data is being used.
        root (str): The root directory of the data files.

    Returns:
        Dict[str, str]: A dictionary with keys as the summary IDs and values as the file paths for each summary in the directory.
    """
    path = f"{root}/data/"
    data_type = "training/" if training else "validation/"
    path += data_type
    path += "gold_summaries/"
    if not os.path.exists(path):
        print(f"Specified path: {path} does not exist")
        return {}

    file_handles = {}
    for f in os.listdir(path):
        report_id = f.split("_")[0]
        summary_id = f.split("_")[1].replace(".txt", "")
        if str(report_id) == str(file_id):
            file_handles[str(summary_id)] = f"{path}{f}"
    return file_handles


def get_company_from_id(
    file_id, training: bool = True, root: str = ".."
) -> Union[str, None]:
    """
    Given a report ID, returns the name of the company that the report is from.

    Args:
        file_id: The ID of the report to find the company name for.
        training (bool): A boolean to indicate whether the training data is being used.
        root (str): The root directory of the data files.

    Returns:
        Union[str, None]: The name of the company or None if it could not be found.
    """
    path = f"{root}/data/"
    data_type = "training/" if training else "validation/"
    path += data_type
    path += "annual_reports/"
    path += str(file_id)
    path += ".txt"
    max_line_iter = (
        100  # assume company name is before the 100th line to save on reading time
    )
    with open(path) as file:
        for i, line in enumerate(file):
            if " plc" in line:
                return clean_company_name(line)
            if i > max_line_iter:
                print("Name not found")
                return None


def get_id_to_company_mapping(training: bool = True) -> Dict[str, str]:
    """
    Returns a dictionary with keys as report IDs and values as the corresponding company names.

    Args:
        training (bool): A boolean to indicate whether the training data is being used.

    Returns:
        Dict[str, str]: A dictionary with keys as report IDs and values as the corresponding company names.
    """
    path = "data/"
    data_type = "training/" if training else "validation/"
    path += data_type
    path += "annual_reports/"
    id_to_company_dict = {}
    for f in os.listdir(path):
        report_id = f.replace(".txt", "")
        id_to_company_dict[report_id] = get_company_from_id(report_id)
    return id_to_company_dict


def get_report(file_id, training: bool = True, root: str = "..") -> str:
    """
    Given a report ID, returns the full text of the report.

    Args:
        file_id: The ID of the report to retrieve.
        training (bool): A boolean to indicate whether the training data is being used.
        root (str): The root directory of the data files.

    Returns:
        str: The full text of the report.
    """
    path = f"{root}/data/"
    data_type = "training/" if training else "validation/"
    path += data_type
    path += "annual_reports/"
    path += str(file_id) + ".txt"
    f = open(path, "r")
    return f.read()


def get_summary(
    file_id, summary_id=None, training: bool = True, root: str = ".."
) -> str:
    """
    Given a report ID and a summary ID (or None), returns the text of the corresponding summary.

    Args:
        file_id: The ID of the report to find the summary for.
        summary_id: The ID of the summary to retrieve or None to select a random summary.
        training (bool): A boolean to indicate whether the training data is being used.
        root (str): The root directory of the data files.

    Returns:
        str: The text of the summary.
    """
    file_handles = get_gold_summaries_file_handles(file_id, training, root=root)
    if summary_id:
        path = file_handles[str(summary_id)]
    else:
        # select random summary
        path = random.choice(list(file_handles.values()))
    with open(path, "r") as f:
        summary = f.read()
    return summary


def get_all_summaries(
    file_id, training: bool = True, root: str = ".."
) -> Dict[str, str]:
    """
    Given a report ID, returns a dictionary containing all summaries for the report.

    Args:
        file_id (str): The ID of the report to find the summaries for.
        training (bool, optional): A boolean to indicate whether the training data is being used. Defaults to True.
        root (str, optional): The root directory of the data files. Defaults to '..'.

    Returns:
        dict: A dictionary with keys as the summary IDs and values as the file paths for each summary in the directory.
    """
    file_handles = get_gold_summaries_file_handles(file_id, training, root=root)
    summaries = {}
    for idx, path in file_handles.items():
        with open(path, "r") as f:
            summary = f.read()
        summaries[str(idx)] = summary
    return summaries


def get_report_sentences_binary_labels_from_str(
    file_id,
    training: bool = True,
    all_summaries: bool = True,
    summary_id=None,
    root: str = "..",
):
    """
         Provide a mapping between each sentence and its binary label of whether \
            it is part of all/any summary/ies
    """
    report = get_report(file_id=file_id, training=training, root=root)
    if not all_summaries and summary_id is not None:
        gold_summaries = get_summary(file_id=file_id, summary_id=summary_id, root=root)
    else:
        gold_summaries = get_all_summaries(
            file_id=file_id, training=training, root=root
        )
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


def get_binary_labels_data_dir(
    training: bool = True, gold: bool = False, root: str = ".."
):
    file_path = f"{root}/data/"
    if training:
        file_path += "training_binary/"
    else:
        file_path += "validation_binary/"
    if gold:
        file_path += "gold_summaries/"
    else:
        file_path += "annual_reports/"
    os.makedirs(file_path, exist_ok=True)
    return file_path


def get_file_handles_binary(training: bool = True, root: str = "..") -> Dict[str, str]:
    """
        Retrieve file handles for training and validation report sentences and binary labels
    :param training: bool
    :param root: str
    :return: dict of file paths
    """
    path = get_binary_labels_data_dir(training=training, root=root)
    if not os.path.exists(path):
        print(f"Specified path: {path} does not exist")
        return {}
    file_handles = {}
    for f in os.listdir(path):
        report_id = f.replace(".csv", "")
        file_handles[report_id] = f"{path}{f}"
    return file_handles


def generate_binary_labels_for_data(
    training: bool = True, gold: bool = False, root: str = ".."
):
    """
    Create a csv file containing a mapping between preprocessed sentences and a binary score which represents \
    whether the sentence is found in the gold summaries. Compute simple statistics like word count
    """
    for file_path in tqdm(get_file_handles(training=training, gold=gold, root=root)):
        file_id = file_path.split("/")[-1]
        try:
            binary_file_path = (
                get_binary_labels_data_dir(training=training, gold=gold, root=root)
                + file_id
                + ".csv"
            )
            if not os.path.exists(binary_file_path):
                sent_label_mapping = get_report_sentences_binary_labels_from_str(
                    file_id=file_id, training=training, all_summaries=True, root=root
                )
                sent_label_mapping_df = (
                    pd.DataFrame()
                    .from_dict(sent_label_mapping, orient="index")
                    .reset_index()
                )
                sent_label_mapping_df.columns = ["sent", "label"]
                # Add extra columns to handle imbalanced data
                sent_label_mapping_df.reset_index(inplace=True)
                sent_label_mapping_df.rename(
                    columns={"index": "sent_num"}, inplace=True
                )
                sent_label_mapping_df["report"] = int(file_id)
                # Compute simple statistics
                sent_label_mapping_df["words_count"] = sent_label_mapping_df[
                    "sent"
                ].apply(lambda x: len(x.split()) + 1)
                sent_label_mapping_df.to_csv(binary_file_path, index=False)
        except Exception as e:
            print("------------------------------------------")
            print(f"{str(e)} at file {file_id}")
            print("------------------------------------------")


def get_sentence_level_df(
    training: bool = True,
    store_df: bool = True,
    file_path: str = None,
    root: str = "..",
) -> pd.DataFrame:
    """
    Return a dataframe containing sentence rows for all reports.
    """
    data_type = "training" if training else "validation"
    if file_path is not None and os.path.exists(file_path):
        print(f"Loading assembled {data_type} dataframe from {file_path}")
        return pd.read_csv(file_path)
    files = get_file_handles_binary(training=training, root=root)
    dfs = []
    for file_id in tqdm(
        files.keys(), f"Assembling binary classification {data_type} file"
    ):
        path = (
            get_binary_labels_data_dir(training=training, root=root) + file_id + ".csv"
        )
        df = pd.read_csv(path)
        # Ensure the report id exists to partition data before model training
        if "report" not in df.columns:
            df["report"] = int(file_id)
        dfs.append(df)
    dfs = pd.concat(dfs)
    if store_df:
        if file_path is None:
            t = datetime.now().strftime("%Y-%m-%d %H-%M")
            file_path = f"{root}/tmp/{data_type}_corpus_{t}.csv"
        print(f"Storing assembled {data_type} dataframe to {file_path}")
        dfs = dfs.reset_index(drop=True)
        dfs.to_csv(file_path, index=False)
    return dfs


def get_raw_data_dir(training: bool = True, root: str = "..") -> str:
    path = f"{root}/data/"
    data_type = "training/" if training else "validation/"
    path += data_type
    path += "annual_reports/"
    return path


def get_corpus_vocabulary(
    training: bool = True,
    validation: bool = True,
    save_file: bool = True,
    root: str = "..",
    file_path: str = None,
    verbose: bool = False,
) -> List[str]:
    default_file_path = f"{root}/tmp/corpus.txt"
    if file_path is None:
        file_path = default_file_path
    if os.path.exists(file_path):
        print(f"Loading corpus of unique words from {file_path}")
        with open(file_path, "r") as f:
            corpus_str = f.read()
        return corpus_str.split("\n")
    print(f"Assembling corpus of unique words to be stored at {file_path}")
    corpus = set([])
    if training:
        for file_id in tqdm(
            get_file_handles(training=True, root=root).keys(),
            "Retrieving training data",
        ):
            if verbose:
                company = get_company_from_id(file_id=file_id, training=True, root=root)
                print(company, end="\n\n")
            report = get_report(file_id=file_id, training=True, root=root)
            # As classification model works on sentence-level embedding vectors
            # apply word tokenization also on sentence level
            _, report_tokenized = preprocess(report)
            for s in report_tokenized.sentences:
                for t in word_tokenize(s):
                    corpus.add(t)
    if validation:
        for file_id in tqdm(
            get_file_handles(training=False, root=root).keys(),
            "Retrieving validation data",
        ):
            if verbose:
                company = get_company_from_id(
                    file_id=file_id, training=False, root=root
                )
                print(company, end="\n\n")
            report = get_report(file_id=file_id, training=False, root=root)
            # As classification model works on sentence-level embedding vectors
            # apply word tokenization also on sentence level
            _, report_tokenized = preprocess(report)
            for s in report_tokenized.sentences:
                for t in word_tokenize(s):
                    corpus.add(t)
    corpus_arr = sorted(list(corpus))
    if save_file:
        if file_path is None:
            file_path = default_file_path
        with open(f"{file_path}", "w") as f:
            for token in corpus_arr:
                f.write(f"{token}\n")
    return corpus_arr


def get_keyed_word_vectors_pickle(
    embedding_weights=None,
    corpus_file_path: str = None,
    save_file: bool = True,
    root: str = "..",
    file_path: str = None,
):
    """
    Given a corpus, generates an embedding dictionary with token as the key and embedding as value.

    Args:
    - embedding_weights: a dict containing the embeddings for the vocabulary.
    - corpus_file_path: str, path to the corpus file.
    - save_file: bool, whether to save the embedding dictionary as a pickle file.
    - root: str, the root directory to look for the corpus file and save the embedding dictionary.
    - file_path: str, the path to save the embedding dictionary pickle file.

    Returns:
    - token2embedding: a dictionary containing the embeddings for the vocabulary, with token as key.
    """
    # Try directly loading existing embedding dict from pickle file
    default_file_path = f"{root}/tmp/corpus_embeddings.pickle"
    if file_path is None:
        file_path = default_file_path
    if os.path.exists(file_path):
        print(f"Loading Keyed Word Vectors from {file_path}")
        with open(file_path, "rb") as handle:
            token2embedding = pickle.load(handle)
        return token2embedding
    # Or pull corpus and re-generate the embedding dict
    print(f"Loading corpus to re-generate embedding dict")
    tokens = get_corpus_vocabulary(file_path=corpus_file_path, root=root)
    token2embedding = {}
    for token in tokens:
        token2embedding[token] = embedding_weights[token]
    if save_file:
        if file_path is None:
            file_path = default_file_path
        print(f"Saving embedding dict to {file_path}")
        with open(file_path, "wb") as handle:
            pickle.dump(token2embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return token2embedding


def get_embedding_model(root: str = ".."):
    path = f"{root}/resources/FinText_FastText_CBOW/Word_Embedding_2000_2015"
    print(f"Loading Embedding Model from {path}")
    embedding_model = FastText.load(path)
    return embedding_model


def recalc_keyed_vector(
    embedding_weights,
    train_dataloader,
    validation_dataloader,
    file_path: str = None,
    save_file: bool = True,
    root: str = "..",
):
    """
    Recalculates keyed word vectors and saves them to a pickle file.
    If the pickle file already exists, it loads the keyed word vectors from the file.

    Args:
        embedding_weights: A dictionary containing embeddings for the vocabulary of the original embedding.
        train_dataloader: A PyTorch DataLoader object containing training data.
        validation_dataloader: A PyTorch DataLoader object containing validation data.
        file_path: The file path to save the keyed word vectors pickle file. Default is {root}/tmp/corpus_embeddings_CSF.pickle.
        save_file: If True, saves the keyed word vectors to a pickle file. Default is True.
        root: The root directory. Default is '..'.

    Returns:
        token2embedding: A dictionary containing keyed word vectors.
    """
    # Try directly loading existing embedding dict from pickle file
    default_file_path = f"{root}/tmp/corpus_embeddings_CSF.pickle"
    if file_path is None:
        file_path = default_file_path
    if os.path.exists(file_path):
        print(f"Loading Keyed Word Vectors from {file_path}")
        with open(file_path, "rb") as handle:
            token2embedding = pickle.load(handle)
        return token2embedding
    print("Pulling out vocabulary")
    corpus = set()
    for i, (sents, labels) in enumerate(train_dataloader):
        for j, s in enumerate(sents):
            for t in word_tokenize(s):
                corpus.add(t)
    for i, (sents, labels) in enumerate(validation_dataloader):
        for j, s in enumerate(sents):
            for t in word_tokenize(s):
                corpus.add(t)
    tokens = sorted(list(corpus))
    print("Recalculating Keyed Word Vector Embeddings")
    token2embedding = {}
    for token in tokens:
        token2embedding[token] = embedding_weights[token]
    if save_file:
        if file_path is None:
            file_path = default_file_path
        print(f"Saving embedding dict to {file_path}")
        with open(file_path, "wb") as handle:
            pickle.dump(token2embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return token2embedding


def binary_classification_data_preparation(root: str = ".."):
    """
    Transform annual reports and summaries into files for sentence-level binary classification.
    """
    generate_binary_labels_for_data(training=True, root=root)
    generate_binary_labels_for_data(training=False, root=root)
    get_sentence_level_df(training=True, root=root)
    get_sentence_level_df(training=False, root=root)
    get_corpus_vocabulary(root=root)
    embedding_model = get_embedding_model(root=root)
    embedding_weights = embedding_model.wv
    get_keyed_word_vectors_pickle(embedding_weights=embedding_weights, root=root)
