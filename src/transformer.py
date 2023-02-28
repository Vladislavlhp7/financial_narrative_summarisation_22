import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments


def load_data(tokenizer, root: str = '..'):
    df = pd.read_csv(f'{root}/tmp/training_corpus_2023-02-07 16-33.csv')
    df_train, df_val = train_test_split(df, stratify=df['label'], test_size=0.1, random_state=42)
    df_test = pd.read_csv(f'{root}/tmp/validation_corpus_2023-02-07 16-33.csv')

    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)
    dataset_test = Dataset.from_pandas(df_test)

    dataset_train = dataset_train.map(
        lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=128), batched=True)
    dataset_val = dataset_val.map(lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=128),
                                  batched=True)
    dataset_test = dataset_test.map(
        lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=128), batched=True)

    dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    return dataset_train, dataset_val, dataset_test


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(predictions, labels)}


def run_experiment(config=None, root: str = '..'):
    torch.cuda.is_available()

    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
    dataset_train, dataset_val, dataset_test = load_data(tokenizer=tokenizer, root=root)

    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain', num_labels=2)
    model_name = 'finbert-sentiment-epoch-1/'

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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.eval()

    metrics = trainer.predict(dataset_test).metrics
    print(metrics)

    trainer.save_model(model_name)


def main():
    run_experiment()


main()
