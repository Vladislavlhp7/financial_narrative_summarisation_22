import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments

from extractor import FNS2021
from metrics import binary_classification_metrics


def load_data(tokenizer, root: str = '..', seed: int = 42, training_downsample_rate: float = 0.9, data_augmentation='fr'):
    print('Loading Training Data')
    data_filename = 'training_corpus_2023-02-07 16-33.csv'
    training_data = FNS2021(file=f'{root}/tmp/{data_filename}', type_='training', random_state=seed,
                            downsample_rate=training_downsample_rate,
                            data_augmentation=data_augmentation)  # aggressive downsample
    validation_data = FNS2021(file=f'{root}/tmp/{data_filename}', type_='validation', random_state=seed,
                              downsample_rate=None)  # use all validation data
    df_train, df_val = training_data.sent_labels_df, validation_data.sent_labels_df

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
    pred_labels, true_labels = eval_pred
    pred_labels = np.argmax(pred_labels, axis=1)
    return binary_classification_metrics(true_labels=true_labels, pred_labels=pred_labels)


def run_experiment(root: str = '..', seed: int = 42, data_augmentation='fr'):
    torch.cuda.is_available()

    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
    dataset_train, dataset_val, dataset_test = load_data(tokenizer=tokenizer, root=root, seed=seed,
                                                         data_augmentation=data_augmentation)

    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain', num_labels=2)
    run_name = 'extractive_summarisation'
    model_name = 'finbert-sentiment-epoch-3/'

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
        metric_for_best_model='accuracy',
        run_name=run_name,
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
    with torch.no_grad():
        prediction_obj = trainer.predict(dataset_test)
        metrics = prediction_obj.metrics
        print(metrics)
        wandb.log(metrics)
    trainer.save_model(model_name)


def main():
    seed = 42
    data_augmentation = 'fr'
    run_experiment(seed=seed, data_augmentation=data_augmentation)


main()
