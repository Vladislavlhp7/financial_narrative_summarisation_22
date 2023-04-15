import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from rouge import Rouge
from scipy.special import softmax
from tqdm import tqdm
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, BertConfig, TrainingArguments

from baseline_summarizers import get_baseline_summary
from extractor import batch_str_to_batch_tensors, FinRNN
from metrics import calc_rouge
from preprocessing import clean
from query import get_all_summaries, get_embedding_model, get_report

args = TrainingArguments(
    output_dir='../tmp/',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)


def trim_string(text, max_length=1000):
    words = text.split()
    if len(words) > 1000:
        return ' '.join(words[:max_length])
    else:
        return text


def rouge_max_top_n_sentences(summary_sentences, reference_summary: str, n: int = 40):
    """
    Extract the top-N sentences from a list of summary sentences based on their ROUGE-2 scores and sort them in their natural order.

    Parameters:
    -----------
    summary_sentences (List[str]): A list of candidate summary sentences.
    reference_summary (str): The reference document summary used to compute the ROUGE-2 scores.
    n (int, optional): The number of top sentences to extract (default: 40).

    Returns:
    --------
    List[str]: A list of the top-N summary sentences sorted in their natural order.
    """
    # Initialize the Rouge object
    rouge = Rouge()

    # Compute the ROUGE-2 scores for each sentence in summary_sentences
    rouge_2_scores = [rouge.get_scores(sentence, reference_summary, avg=True)["rouge-2"]["f"] for sentence in summary_sentences]

    # Get the indices that would sort the sentences by their ROUGE-2 scores in descending order
    sorted_indices = np.argsort(rouge_2_scores)[::-1]

    # Select the top-N sentence indices based on the ROUGE-2 scores
    top_n_indices = sorted_indices[:n]

    # Re-sort the top-N sentence indices in their natural order
    top_n_indices_natural_order = sorted(top_n_indices)

    # Extract the top-N summary sentences in their natural order
    summary_sentences_top = [summary_sentences[i] for i in top_n_indices_natural_order]

    return summary_sentences_top


def select_summary_sents(probabilities, sentences, max_sents=int(1000/25), rouge_maximisation=False, reference_summary=None):
    """
    Selects a summary of the input sentences based on the predicted probabilities of label 1.

    Args:
        probabilities (numpy.ndarray): Array of predicted probabilities with shape (N, 2).
        sentences (List[str]): List of input sentences.
        max_sents (int, optional): Maximum number of sentences to include in the summary. Defaults to 40.
        rouge_maximisation (bool, optional): Whether to maximise sentence-level ROUGE-2 score. Defaults to True.
        reference_summary (str, optional): Reference summary to use for ROUGE-2 maximisation. Defaults to None.

    Returns:
        summary (str): The selected summary.
    """
    # Convert the numpy array to a PyTorch tensor
    predictions = torch.tensor(probabilities)
    # Get the indices of rows where label 1 is predicted
    summary_indices = torch.where(predictions[:, 1] >= 0.5)[0]
    # Filter the tensor for rows where label 1 is predicted
    summary_probs = predictions[summary_indices]
    summary_sentences = [sentences[i] for i in summary_indices]
    if rouge_maximisation and reference_summary is not None:
        summary_sentences_top = rouge_max_top_n_sentences(summary_sentences, reference_summary, n=min(max_sents, len(summary_sentences)))
    else:
        # Get the top-k probabilities and corresponding sentence indices
        summary_probs_top, summary_indices_top = torch.topk(summary_probs[:, 1], k=min(max_sents, summary_probs.shape[0]))
        # Get the top-k sentences based on the selected indices
        summary_sentences_top = [summary_sentences[i] for i in summary_indices_top]
    # Join the selected sentences to form the summary
    summary = " ".join(summary_sentences_top)
    # Ensure that the summary is less than 1000 words
    summary = trim_string(summary)
    return summary


def load_data_transformer(tokenizer, df_test):
    dataset_test = Dataset.from_pandas(df_test)
    dataset_test = dataset_test.map(
        lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=128), batched=True)
    dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    return dataset_test


def get_max_dict(dict_list):
    """
    Returns the dictionary from a list of dictionaries with the maximal 'f' value,
    where each dictionary corresponds to a document summary evaluated using the
    ROUGE evaluation metric.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics for
    evaluating automatic summarization and machine translation software. The metrics
    compare an automatically generated summary or translation against a reference
    summary or translation, and compute measures of overlap, such as precision,
    recall, and F-measure.

    Parameters:
    -----------
    dict_list: list of dict
        A list of dictionaries, where each dictionary has a single key-value pair,
        where the value is a dictionary with a single key 'f' and a numeric value.

    Returns:
    --------
    max_dict: dict or None
        The dictionary from `dict_list` with the maximal 'f' value. If `dict_list` is
        empty, returns `None`.
    """
    max_dict = None
    max_f = -float('inf')
    for d in dict_list:
        f = d[list(d.keys())[0]]['f']
        if f > max_f:
            max_dict = d
            max_f = f
    return max_dict


def get_max_rouge_l_score(data):
    max_score = 0
    max_dict = {}
    for d in data:
        if d['rouge-l']['f'] > max_score:
            max_score = d['rouge-l']['f']
            max_dict = d
    return max_dict


def load_model_transformer(model_dir, device='cpu'):
    output_model_file = f"{model_dir}/pytorch_model.bin"
    output_config_file = f"{model_dir}/config.json"

    config = BertConfig.from_json_file(output_config_file)
    model_transformer = BertForSequenceClassification(config)

    state_dict = torch.load(output_model_file, map_location=torch.device(device))
    model_transformer.load_state_dict(state_dict, )
    return model_transformer


def evaluate_model(model, config, embedding_model=None, rouge_maximisation=False):
    """
        Evaluates the model on the test set. The model is evaluated using the ROUGE evaluation metric.
    """
    trainer = None
    tokenizer = None
    predictions = None
    sentences = None
    if config['model_type'] == 'transformer':
        trainer = Trainer(model=model, args=args)

    model.eval()
    df_test = pd.read_csv(f'{config["df_test_path"]}')
    report_ids = df_test['report'].unique()
    rouge_scores = []
    for report_id in tqdm(report_ids):
        df_test_report = df_test.loc[df_test.report.isin([int(report_id), str(report_id)])]
        sentences = df_test_report['sent'].tolist()
        # labels = df_test_report['label'].tolist()
        if config['model_type'] == 'gru':
            inputs_embedded = batch_str_to_batch_tensors(sentence_list=sentences, embedding_model=embedding_model)
            with torch.no_grad():
                outputs = model(inputs_embedded)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            predictions = outputs.numpy()
        elif config['model_type'] == 'transformer':
            tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
            inputs_embedded = load_data_transformer(tokenizer=tokenizer, df_test=df_test_report)
            outputs = trainer.predict(inputs_embedded)
            print(outputs)
            predictions = softmax(outputs.predictions, axis=1)
            print(predictions)

        # Calculate ROUGE scores for each summary compared to the generated summary
        rouge_scores_per_summary = []
        gold_summaries_dict = get_all_summaries(file_id=report_id, training=False)
        for _, gold_summary in gold_summaries_dict.items():
            try:
                gold_summary_ = clean(gold_summary).lower()
                generated_summary = select_summary_sents(predictions, sentences, rouge_maximisation=rouge_maximisation, reference_summary=gold_summary_)
                rouge_score = calc_rouge(generated_summary, gold_summary_)
            except ValueError:
                rouge_score = {'rouge-l': {'f': 0}, 'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}}
            rouge_scores_per_summary.append(rouge_score)
        # Get the maximum ROUGE-L score from the list of scores per summary
        max_rouge_l_score = get_max_rouge_l_score(rouge_scores_per_summary)
        # Append the max ROUGE-L score to the list of scores for all reports
        rouge_scores.append(max_rouge_l_score)
    return rouge_scores


def generate_summaries(model, config, embedding_model=None, num_reports=10):
    """
        Evaluates the model on the test set. The model is evaluated using the ROUGE evaluation metric.
    """
    trainer = None
    tokenizer = None
    predictions = None
    sentences = None
    if config['model_type'] == 'transformer':
        trainer = Trainer(model=model, args=args)

    model.eval()
    df_test = pd.read_csv(f'{config["df_test_path"]}')
    report_ids = df_test['report'].unique()
    generated_summaries = []
    for report_id in tqdm(report_ids[:num_reports]):
        df_test_report = df_test.loc[df_test.report.isin([int(report_id), str(report_id)])]
        sentences = df_test_report['sent'].tolist()
        if config['model_type'] == 'gru':
            inputs_embedded = batch_str_to_batch_tensors(sentence_list=sentences, embedding_model=embedding_model)
            with torch.no_grad():
                outputs = model(inputs_embedded)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            predictions = outputs.numpy()
        elif config['model_type'] == 'transformer':
            tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
            inputs_embedded = load_data_transformer(tokenizer=tokenizer, df_test=df_test_report)
            outputs = trainer.predict(inputs_embedded)
            print(outputs)
            predictions = softmax(outputs.predictions, axis=1)
            print(predictions)
        generated_summary = select_summary_sents(predictions, sentences, rouge_maximisation=False)
        generated_summaries.append(str(generated_summary))
    return generated_summaries


def evaluate_baseline(config, max_sents=int(1000/25)):
    df_test = pd.read_csv(f'{config["df_test_path"]}')
    reports = df_test['report'].unique()
    rouge_scores = []
    for report_id in tqdm(reports):
        report = get_report(report_id, training=False)
        generated_summary = get_baseline_summary(text=clean(report), max_sents=max_sents, method=config['model_name'])

        # Calculate ROUGE scores for each summary compared to the generated summary
        rouge_scores_per_summary = []
        gold_summaries_dict = get_all_summaries(file_id=report_id, training=False)
        for _, gold_summary in gold_summaries_dict.items():
            gold_summary_ = clean(gold_summary).lower()
            rouge_score = calc_rouge(generated_summary, gold_summary_)
            rouge_scores_per_summary.append(rouge_score)
        # Get the maximum ROUGE-L score from the list of scores per summary
        max_rouge_l_score = get_max_rouge_l_score(rouge_scores_per_summary)

        # Append the max ROUGE-L score to the list of scores for all reports
        rouge_scores.append(max_rouge_l_score)
    return rouge_scores


def evaluate_baselines(configs, max_sents=int(1000/25)):
    rouge_scores_list = []
    for c in tqdm(configs):
        rouge_scores = evaluate_baseline(config=c, max_sents=max_sents)
        print(rouge_scores)
        rouge_scores_list.append(rouge_scores)
        df = rouge_dict_to_df(rouge_scores)
        df.to_csv(f"{c['model_name']}_rouge_scores.csv")
        # Save the list of dictionaries to a file
        with open(f"{c['model_name']}_rouge_scores.json", "w") as outfile:
            json.dump(rouge_scores_list, outfile)


def unpack_rouge_metric(d, key='f'):
    """
        Unpacks the 'f' key from the ROUGE dictionary.

        Args:
        - d: a dictionary containing ROUGE scores
        - key: the key to unpack, default is 'f'

        Returns:
        - a dictionary with the 'f' key unpacked
    """
    d_new = {k: v[key] for k, v in d.items() if key in v}
    return d_new


def rouge_dict_to_df(data):
    """
        Converts a list of ROUGE dictionaries into a pandas DataFrame.

        Args:
        - data: a list of dictionaries containing ROUGE scores

        Returns:
        - a pandas DataFrame with ROUGE scores as columns and each row corresponding to a single document
    """
    data_ = []
    for i, d in enumerate(data):
        d_new = unpack_rouge_metric(d)
        data_.append(d_new)
    df = pd.DataFrame(data_)
    return df


def evaluate_models(configs, embedding_model=None, rouge_maximisation=True, device='cpu'):
    rouge_scores_list = []
    for c in tqdm(configs, desc='Evaluating models'):
        print(f"{c['model_type']}")
        rouge_scores = None
        if c['model_type'] == 'transformer':
            model_transformer = load_model_transformer(c['model_dir'], device=device)
            rouge_scores = evaluate_model(config=c, model=model_transformer, embedding_model=None, rouge_maximisation=rouge_maximisation)
        elif c['model_type'] == 'gru':
            model_rnn = FinRNN(hidden_size=c['hidden_size'])
            model_rnn.load_state_dict(torch.load(c['model_path'], map_location=torch.device(device)), strict=False)
            rouge_scores = evaluate_model(config=c, model=model_rnn, embedding_model=embedding_model, rouge_maximisation=rouge_maximisation)
        print(rouge_scores)
        rouge_scores_list.append(rouge_scores)
        df = rouge_dict_to_df(rouge_scores)
        df.to_csv(f"{c['model_name']}_rouge_scores{str(rouge_maximisation)}.csv")
        # Save the list of dictionaries to a file
        with open(f"{c['model_name']}_rouge_scores{str(rouge_maximisation)}.json", "w") as outfile:
            json.dump(rouge_scores_list, outfile)


def main():
    embedding_model = get_embedding_model()
    # Set device to CPU or CUDA
    torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = []
    # GRU models
    # 64-0.9-None-None
    config = {'model_type': 'gru', 'df_test_path': '../tmp/validation_corpus_2023-02-07 16-33.csv', 'hidden_size': 64,
              'model_name': 'model-0.001-64-0.9-2023-04-12-18-30.h5'}
    config['model_path'] = config['model_name']
    configs.append(config)

    # 64-0.9-None-dot
    config = {'model_type': 'gru', 'df_test_path': '../tmp/validation_corpus_2023-02-07 16-33.csv', 'hidden_size': 64,
              'model_name': 'model-0.001-64-0.9-2023-04-12-17-05.h5'}
    config['model_path'] = config['model_name']
    configs.append(config)

    # 64-0.8-fr-None
    config = {'model_type': 'gru', 'df_test_path': '../tmp/validation_corpus_2023-02-07 16-33.csv', 'hidden_size': 64,
              'model_name': 'model-0.001-64-0.8-2023-04-13-04-12.h5'}
    config['model_path'] = config['model_name']
    configs.append(config)

    # 64-0.8-fr-dot
    config = {'model_type': 'gru', 'df_test_path': '../tmp/validation_corpus_2023-02-07 16-33.csv', 'hidden_size': 64,
              'model_name': 'model-0.001-64-0.8-2023-04-12-21-08.h5'}
    config['model_path'] = config['model_name']
    configs.append(config)

    # Transformer model
    config = {
        'seed_v': 42,
        'data_augmentation': 'fr',
        'lr': 2e-5,
        'training_downsample_rate': 0.8,
        'model_type': 'transformer',
        'df_test_path': '../tmp/validation_corpus_2023-02-07 16-33.csv',
    }
    config['model_name'] = f"finbert-sentiment-seed-{config['seed_v']}-dataaugm-{config['data_augmentation']}-lr-{config['lr']}-downsample-{config['training_downsample_rate']}"
    config['model_dir'] = config['model_name']
    configs.append(config)

    config = {
        'seed_v': 42,
        'data_augmentation': None,
        'lr': 2e-5,
        'training_downsample_rate': 0.9,
        'model_type': 'transformer',
        'df_test_path': '../tmp/validation_corpus_2023-02-07 16-33.csv',
    }
    config[
        'model_name'] = f"finbert-sentiment-seed-{config['seed_v']}-dataaugm-{config['data_augmentation']}-lr-{config['lr']}-downsample-{config['training_downsample_rate']}"
    config['model_dir'] = config['model_name']
    configs.append(config)

    evaluate_models(configs=configs, embedding_model=embedding_model, rouge_maximisation=True, device=device)

    # configs = []
    # config = {
    #     'model_name': 'lex',
    #     'df_test_path': '../tmp/validation_corpus_2023-02-07 16-33.csv',
    # }
    # configs.append(config)
    # config = {
    #     'model_name': 'textrank',
    #     'df_test_path': '../tmp/validation_corpus_2023-02-07 16-33.csv',
    # }
    # configs.append(config)
    # evaluate_baselines(configs)


if __name__ == '__main__':
    main()
