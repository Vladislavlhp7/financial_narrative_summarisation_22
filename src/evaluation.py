import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, BertConfig, TrainingArguments
from datasets import Dataset
from extractor import FinRNN
from query import get_embedding_model
from extractor import batch_str_to_batch_tensors


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
    summary_probs_top, summary_indices_top = torch.topk(summary_probs[:, 1], k=max_sents)
    # Get the top-k sentences based on the selected indices
    summary_sentences_top = [summary_sentences[i] for i in summary_indices_top]
    # Join the selected sentences to form the summary
    summary = " ".join(summary_sentences_top)
    # Ensure that the summary is less than 1000 words
    summary = trim_string(summary)
    return summary
