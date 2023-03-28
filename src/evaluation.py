import pandas as pd
import torch
from datasets import Dataset
from scipy.special import softmax
from tqdm import tqdm
from transformers import BertTokenizer, Trainer, TrainingArguments

from extractor import batch_str_to_batch_tensors
from metrics import calc_rouge
from preprocessing import clean
from query import get_all_summaries

args = TrainingArguments(
        output_dir='../tmp/',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
)


def trim_string(text, max_length=1000):
    words = text.split()
    if len(words) > 1000:
        return ' '.join(words[:max_length])
    else:
        return text


def select_summary_sents(probabilities, sentences, max_sents=25):
    """
    Selects a summary of the input sentences based on the predicted probabilities of label 1.

    Args:
        probabilities (numpy.ndarray): Array of predicted probabilities with shape (N, 2).
        sentences (List[str]): List of input sentences.
        max_sents (int, optional): Maximum number of sentences to include in the summary. Defaults to 25.

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
    dataset_test = dataset_test.map(lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=128), batched=True)
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


def evaluate_model(model, config, embedding_model=None):
    """
        Evaluates the model on the test set. The model is evaluated using the ROUGE evaluation metric.
    """
    model.eval()

    df_test = pd.read_csv(f'{config["df_test_path"]}')
    reports = df_test['report'].unique()
    rouge_scores = []
    for report in tqdm(reports):
        df_test_report = df_test[df_test['report'] == report]
        sentences = df_test_report['sent'].tolist()
        # labels = df_test_report['label'].tolist()
        inputs_embedded = None
        if config['model_type'] == 'gru':
            inputs_embedded = batch_str_to_batch_tensors(sentence_list=sentences, embedding_model=embedding_model)
        elif config['model_type'] == 'transformer':
            tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
            inputs_embedded = load_data_transformer(tokenizer=tokenizer, df_test=df_test_report)

        with torch.no_grad():
            if config['model_type'] == 'gru':
                outputs = model(inputs_embedded)
                predictions = outputs.numpy()
            elif config['model_type'] == 'transformer':
                trainer = Trainer(
                    model=model,
                    args=args
                )
                outputs = trainer.predict(inputs_embedded)
                predictions = softmax(outputs.predictions, axis=1)
        generated_summary = select_summary_sents(predictions, sentences)

        # Calculate ROUGE scores for each summary compared to the generated summary
        rouge_scores_per_summary = []
        gold_summaries_dict = get_all_summaries(file_id=report, training=False)
        for _, gold_summary in gold_summaries_dict.items():
            rouge_score = calc_rouge(generated_summary, clean(gold_summary).lower())
            rouge_scores_per_summary.append(rouge_score)
        # Get the maximum ROUGE-L score from the list of scores per summary
        max_rouge_l_score = get_max_rouge_l_score(rouge_scores_per_summary)
        # Append the max ROUGE-L score to the list of scores for all reports
        rouge_scores.append(max_rouge_l_score)
    return rouge_scores
