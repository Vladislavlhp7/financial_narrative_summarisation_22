from datetime import datetime

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from query import get_embedding_model, get_keyed_word_vectors_pickle, recalc_keyed_vector


def get_word_embedding(model, word: str):
    """
    Directly return word embedding
    """
    try:  # if loaded directly from embedding model, e.g., FastText
        return model.wv[word]
    except AttributeError:  # if we use a pseudo-model, Keyed Word Vectors over Vocabulary
        return model[word]


def get_sentence_tensor(embedding_model, sentence: str, seq_len: int = 50):
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


def batch_str_to_batch_tensors(train_sents, embedding_model, seq_len: int = 50):
    """
    Convert a list of batch sentences to a batch tensor
    """
    # create a list of word embeddings per sentence
    batch_sent_arr = [get_sentence_tensor(embedding_model=embedding_model,
                                          sentence=str(sent),
                                          seq_len=seq_len) for sent in train_sents]
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
    def __init__(self, file: str, training: bool = True, train_ratio: float = 0.9, random_state: int = 1):
        """
        Custom class for FNS 2021 Competition to load training and validation data. \
        Original validation data is used as testing
        """
        self.total_data_df = pd.read_csv(file)
        train_df, validation_df = train_test_split(self.total_data_df, test_size=1 - train_ratio,
                                                   random_state=random_state, stratify=self.total_data_df.label)
        if training:
            self.sent_labels_df = train_df
        else:
            self.sent_labels_df = validation_df

    def __len__(self):
        return len(self.total_data_df)

    def __getitem__(self, idx):
        sent = self.total_data_df.loc[idx, 'sent']
        label = self.total_data_df.loc[idx, 'label']
        return sent, label


def train_one_epoch(model, train_dataloader, embedding_model, seq_len, epoch_index, writer, criterion, optimizer):
    running_loss = 0.
    last_loss = 0.

    for i, (train_sents, train_labels) in enumerate(train_dataloader):
        batch_sent_tensor = batch_str_to_batch_tensors(train_sents=train_sents, embedding_model=embedding_model,
                                                       seq_len=seq_len)
        train_labels = train_labels.long()

        optimizer.zero_grad()
        output_labels = model(batch_sent_tensor)
        loss = criterion(output_labels, train_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
            torch.save(model.state_dict(), model.name)
    return last_loss


def train(model, embedding_model, train_dataloader, validation_dataloader, writer,
          epochs: int = 60, lr: float = 1e-3, seq_len: int = 50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyTrainingStop()

    for epoch in tqdm(range(epochs)):
        print('EPOCH {}:'.format(epoch + 1))
        # Training 1 Epoch
        model.train(True)
        training_loss = train_one_epoch(model=model, embedding_model=embedding_model, seq_len=seq_len,
                                        epoch_index=epoch, writer=writer, criterion=criterion,
                                        optimizer=optimizer, train_dataloader=train_dataloader)
        # Validation
        model.train(False)
        running_vloss = 0.
        i = 0
        for i, (v_sents, v_labels) in enumerate(validation_dataloader):
            batch_sent_tensor = batch_str_to_batch_tensors(train_sents=v_sents, embedding_model=embedding_model,
                                                           seq_len=seq_len)
            train_labels = v_labels.long()

            output_labels = model(batch_sent_tensor)
            vloss = criterion(output_labels, train_labels)
            running_vloss += vloss
        validation_loss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(training_loss, validation_loss))
        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': training_loss, 'Validation': validation_loss}, epoch + 1)
        writer.flush()
        # Stop training if validation loss starts growing and save model parameters
        if early_stopper.early_stop(validation_loss=validation_loss):
            torch.save(model.state_dict(), model.name)
            break


def run(root: str = '..', batch_size: int = 16, EPOCHS: int = 3, lr: float = 1e-3):
    REGEN_VOCAB = False
    LOAD_KEYED_VECTOR = True
    input_size = 300
    seq_len = 50
    num_layers = 2

    # Set device to CPU or CUDA
    cuda = torch.cuda.is_available()
    if cuda:
        print('Computational device chosen: CUDA')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print('Computational device chosen: CPU')

    print('Loading Training & Validation Data')
    data_filename = 'training_corpus_2023-02-07 16-33.csv'
    training_data = FNS2021(file=f'{root}/tmp/{data_filename}', training=True)
    validation_data = FNS2021(file=f'{root}/tmp/{data_filename}', training=False)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, drop_last=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, drop_last=True)

    embedding_model = None
    if REGEN_VOCAB:
        embedding_model_weights = get_embedding_model(root=root).wv
        embedding_model = recalc_keyed_vector(root=root, train_dataloader=train_dataloader,
                                              validation_dataloader=validation_dataloader,
                                              embedding_weights=embedding_model_weights,
                                              file_path=f'{root}/tmp/corpus_embeddings_CSF.pickle')

    # Load Embeddings either by vocabulary keyed vector or FastText model
    if LOAD_KEYED_VECTOR:
        embedding_model = get_keyed_word_vectors_pickle(root=root, file_path=f'{root}/tmp/corpus_embeddings_CSF.pickle')
    elif not REGEN_VOCAB:
        embedding_model = get_embedding_model(root=root)
    else:
        pass

    model = LSTM(input_size=input_size, num_layers=num_layers)

    writer = SummaryWriter('PyCharm-' + model.name)

    print('Starting LSTM training')
    train(model=model, embedding_model=embedding_model,
          train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
          lr=lr, epochs=EPOCHS, seq_len=seq_len, writer=writer)


def experiment1(root: str = '..'):
    lr = 1e-3
    EPOCHS = 3
    batch_size = 16
    run(lr=lr, EPOCHS=EPOCHS, batch_size=batch_size, root=root)


def experiment2(root: str = '..'):
    # TODO: multiple learning rates, many epochs
    pass


def experiment3(root: str = '..'):
    # TODO: Adaptive learning rate, many epochs
    pass


def main():
    nltk.download('punkt')
    root = '..'
    experiment1(root=root)


main()
