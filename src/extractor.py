import gc
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from query import get_embedding_model


def get_word_embedding(model, word: str):
    """
    Directly return word embedding
    """
    try:  # if loaded directly from embedding model, e.g., FastText
        return model.wv[word]
    except AttributeError:  # if we use a pseudo-model, Keyed Word Vectors over Vocabulary
        return model[word]


def get_sentence_tensor(embedding_model, sentence: str, seq_len: int = 100):
    """
    Assemble a sentence tensor by directly loading word embeddings from a pre-trained embedding model up to max length
    """
    sent_arr = []
    for i, word in enumerate(word_tokenize(sentence)):
        if i > seq_len:
            break
        sent_arr.append(get_word_embedding(embedding_model, word))
    sent_tensor = torch.FloatTensor(np.array(sent_arr))
    return sent_tensor


class EarlyTrainingStop:
    """
    Implement a class for early stopping of training when validation loss starts increasing
    """

    def __init__(self, validation_loss: float = np.inf, delta: float = 0.0, counter: int = 0, patience: int = 1):
        self.validation_loss = validation_loss
        self.delta = delta
        self.counter = counter
        self.patience = patience

    def early_stop(self, validation_loss: float):
        if self.validation_loss <= validation_loss + self.delta:
            self.counter += 1
            if self.counter > self.patience:
                return True
        else:
            self.counter = 0
            self.validation_loss = validation_loss


# pad a batch of sentence tensors
def pad_batch(batch_sent_arr):
    """
    Provide a batch (list) of tensor sentences and pad them to the maximal size
    Return a batch (list) of same-size sentences
    """
    max_len = max([x.shape[0] for x in batch_sent_arr])
    padded_batch = []
    for train_sents in batch_sent_arr:
        padded_train_sents = torch.zeros(max_len, train_sents.shape[1], dtype=torch.float32)
        padded_train_sents[:train_sents.shape[0]] = train_sents
        padded_batch.append(padded_train_sents)
    return padded_batch


def batch_str_to_batch_tensors(sentence_list, embedding_model, seq_len: int = 100):
    """
    Convert a list of batch sentences to a batch tensor
    """
    # create a list of word embeddings per sentence
    batch_sent_arr = [get_sentence_tensor(embedding_model=embedding_model,
                                          sentence=str(sent),
                                          seq_len=seq_len) for sent in sentence_list]
    # ensure all sentences (tensors) in the batch have the same length, hence padding
    batch_sent_arr_padded = pad_batch(batch_sent_arr)
    # stack sentence tensors onto each other for a batch tensor
    batch_sent_tensor = torch.stack(batch_sent_arr_padded)
    return batch_sent_tensor


class LSTM(nn.Module):
    def __init__(self, input_size=300, hidden_size=256, num_layers=2, label_size=2, bidirectional=True,
                 batch_first=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.name = f'LSTM_bin_classifier-{dt}.pt'
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=batch_first)
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.hidden2label = nn.Linear(in_features=self.D * hidden_size, out_features=label_size)

    def forward(self, sent):
        out, _ = self.lstm(sent)
        out = out[:, -1, :]
        out = self.hidden2label(out)
        return F.softmax(out, dim=1)


class FNS2021(Dataset):
    def __init__(self, file: str, training: bool = True, train_ratio: float = 0.9, random_state: int = 1,
                 downsample_rate: float = None):
        """
        Custom class for FNS 2021 Competition to load training and validation data. \
        Original validation data is used as testing
        """
        self.total_data_df = pd.read_csv(file).drop(columns=['Unnamed: 0'], errors='ignore')
        self.total_data_df.index.name = 'sent_index'
        self.total_data_df.reset_index(inplace=True)
        train_df, validation_df = train_test_split(self.total_data_df, test_size=1 - train_ratio,
                                                   random_state=random_state, stratify=self.total_data_df.label)
        if training:
            if downsample_rate is not None:
                train_df = self.downsample(df=train_df, rate=downsample_rate, random_state=random_state)
            self.sent_labels_df = train_df
        else:
            self.sent_labels_df = validation_df
        self.sent_labels_df.reset_index(drop=True, inplace=True)

    @staticmethod
    def downsample(df: pd.DataFrame, rate: float = 0.5, random_state: int = 1):
        # TODO: Downsample only when report data is predominantly 0-labeled
        summary_df = df.loc[df['label'] == 1]
        non_summary_df = df.loc[df['label'] == 0]
        non_summary_df = resample(non_summary_df,
                                  replace=True,
                                  n_samples=int(len(non_summary_df) * (1 - rate)),
                                  random_state=random_state)
        df = pd.concat([summary_df, non_summary_df]).sort_values(['sent_index'])  # .reset_index(drop=True)
        return df

    def __len__(self):
        return len(self.sent_labels_df)

    def __getitem__(self, idx):
        sent = self.sent_labels_df.loc[idx, 'sent']
        label = self.sent_labels_df.loc[idx, 'label']
        return sent, label


def train_one_epoch(model, train_dataloader, embedding_model, seq_len, epoch, criterion, device,
                    optimizer, save_checkpoint):
    model.train(True)
    running_loss = 0.0
    last_loss = 0.0
    running_acc = 0.0  # Total accuracy for both classes
    running_acc_1 = 0.0  # Accuracy for summary class

    for i, (data, labels) in enumerate(train_dataloader):
        batch_sent_tensor = batch_str_to_batch_tensors(sentence_list=data, embedding_model=embedding_model,
                                                       seq_len=seq_len).to(device)
        target = labels.long().to(device)  # 1-dimensional integer tensor of size d
        predicted = model(batch_sent_tensor)  # (d,2) float probability tensor
        loss = criterion(predicted, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Calculate and record per-batch accuracy
        winners = predicted.argmax(dim=1)  # each sentence has p0 and p1 probabilities with p0 + p1 = 1
        corrects = (winners == target)  # match predicted output labels with observed labels
        accuracy = corrects.sum().float() / float(target.size(0))
        running_acc += accuracy
        summary_winners = ((winners == target) * (target == 1)).float()
        summary_winners_perc = summary_winners.sum() / max((target == 1).sum(), 1)
        running_acc_1 += summary_winners_perc.sum()

        # Increment per-batch loss
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            last_acc = running_acc / 1000  # total accuracy per batch
            last_acc_1 = running_acc_1 / 1000  # summary sent accuracy per batch
            print('  batch {} loss: {} total accuracy: {} summary accuracy: {}, '.format(i + 1, last_loss, last_acc,
                                                                                         last_acc_1))
            # Log training details on W&B
            wandb.log({
                "Training Loss": last_loss,
                "Epoch": epoch,
                "Batch": i + 1,
                "Total Training Accuracy": last_acc,
                "Summary Training Accuracy": last_acc_1,
            })
            running_loss = 0.0
            running_acc = 0.0
            running_acc_1 = 0.0
            if save_checkpoint:
                model.cpu()
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': last_loss}, model.name)
                model.cuda()
    return last_loss


def validate(model, embedding_model, validation_dataloader, criterion, seq_len, device, epoch):
    running_vloss = 0.0
    running_acc = 0.0  # Total accuracy for both classes
    running_acc_1 = 0.0  # Accuracy for summary class
    model.eval()
    with torch.no_grad():
        for i, (v_data, v_labels) in enumerate(validation_dataloader):
            batch_sent_tensor = batch_str_to_batch_tensors(sentence_list=v_data, embedding_model=embedding_model,
                                                           seq_len=seq_len).to(device)
            target = v_labels.long().to(device)
            predicted = model(batch_sent_tensor)
            vloss = criterion(predicted, target)
            # Record accuracy & loss
            running_vloss += vloss
            # Calculate and record per-batch accuracy
            winners = predicted.argmax(dim=1)  # each sentence has p0 and p1 probabilities with p0 + p1 = 1
            corrects = (winners == target)  # match predicted output labels with observed labels
            accuracy = corrects.sum().float() / float(target.size(0))
            running_acc += accuracy
            summary_winners = ((winners == target) * (target == 1)).float()
            summary_winners_perc = summary_winners.sum() / max((target == 1).sum(), 1)
            running_acc_1 += summary_winners_perc.sum()
            if i % 1000 == 999:
                last_loss = running_vloss / 1000  # loss per batch
                last_acc = running_acc / 1000  # total accuracy per batch
                last_acc_1 = running_acc_1 / 1000  # summary sent accuracy per batch
                print('Validation loss {} total accuracy: {} summary accuracy: {}'.format(last_loss, last_acc,
                                                                                          last_acc_1))
                wandb.log({
                    "Epoch": epoch,
                    "Validation Loss": last_loss,
                    "Total Validation Accuracy": last_acc,
                    "Summary Validation Accuracy": last_acc_1,
                })
                running_vloss = 0.0
                running_acc = 0.0
                running_acc_1 = 0.0
    return last_loss


def train_epochs(model, embedding_model, device, optimizer, train_dataloader, validation_dataloader, save_checkpoint,
                 epochs: int = 60, seq_len: int = 100):
    print('Starting model training')
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyTrainingStop()

    for epoch in tqdm(range(epochs)):
        print('EPOCH {}:'.format(epoch))
        training_loss = train_one_epoch(model=model, embedding_model=embedding_model, seq_len=seq_len,
                                        epoch=epoch, criterion=criterion,
                                        save_checkpoint=save_checkpoint, device=device,
                                        optimizer=optimizer, train_dataloader=train_dataloader)
        # Validation
        validation_loss = validate(model=model, embedding_model=embedding_model,
                                   validation_dataloader=validation_dataloader,
                                   criterion=criterion, seq_len=seq_len, device=device, epoch=epoch)
        print('LOSS train {} valid {}'.format(training_loss, validation_loss))
        # Stop training if validation loss starts growing and save model parameters
        if early_stopper.early_stop(validation_loss=validation_loss):
            model.cpu()
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': training_loss,
                'validation_loss': validation_loss}, model.name)
            model.cuda()
            break


def run_experiment(config=None, root: str = '..'):
    save_checkpoint = False
    input_size = 300  # FastText word-embedding dimensions
    seq_len = 100  # words per sentence
    num_layers = 2  # layers of LSTM model

    # Set device to CPU or CUDA
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda:
        print('Computational device chosen: CUDA')
        # Empty CUDA cache
        gc.collect()
        torch.cuda.empty_cache()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Set data types to default CUDA standard
    else:
        print('Computational device chosen: CPU')

    # Load Embeddings directly from FastText model
    embedding_model = get_embedding_model(root=root)

    with wandb.init(resume='allow', project='FNS-biLSTM-classification-sweep', config=config):
        config = wandb.config
        config.test_batch_size = config.batch_size
        torch.manual_seed(1)  # pytorch random seed

        print('Loading Training Data')
        data_filename = 'training_corpus_2023-02-07 16-33.csv'
        training_data = FNS2021(file=f'{root}/tmp/{data_filename}', training=True,
                                downsample_rate=config.downsample_rate)  # aggressive downsample
        train_dataloader = DataLoader(training_data, batch_size=config.batch_size, drop_last=True)
        print('Loading Validation Data')
        validation_data = FNS2021(file=f'{root}/tmp/{data_filename}', training=False,
                                  downsample_rate=None)  # use all validation data
        validation_dataloader = DataLoader(validation_data, batch_size=config.batch_size, drop_last=True)

        model = LSTM(input_size=input_size, num_layers=num_layers, hidden_size=config.hidden_size)
        model_name = f'model-{config.lr}-{config.hidden_size}-{config.downsample_rate}-{datetime.now().strftime("%Y-%m-%d-%H-%M")}.h5'
        model.name = model_name

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        train_epochs(model=model, embedding_model=embedding_model, device=device,
                     train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
                     optimizer=optimizer,
                     epochs=config.epochs, seq_len=seq_len, save_checkpoint=save_checkpoint)
        wandb.save(model_name)
        model.cpu()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model.name)
        model.cuda()
