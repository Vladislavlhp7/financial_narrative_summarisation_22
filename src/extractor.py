import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import FastText
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def get_word_embedding(model, word: str):
    """
    Directly return word embedding
    """
    return model.wv[word]


def get_sentence_tensor(embedding_model, sentence: str, seq_len: int = 50):
    """
    Assemble a sentence tensor by directly loading word embeddings from a pre-trained embedding model up to max length
    """
    sent_arr = []
    for i, word in enumerate(word_tokenize(sentence)):
        if i > seq_len:
            break
        sent_arr.append(get_word_embedding(embedding_model, word))

    sent_tensor = torch.FloatTensor(sent_arr)
    return sent_tensor


# pad a batch of sentences
def pad_batch(train_sents_tensor):
    max_len = max([x.shape[0] for x in train_sents_tensor])
    padded_batch = []
    for train_sents in train_sents_tensor:
        padded_train_sents = torch.zeros(max_len, train_sents.shape[1], dtype=torch.float32)
        padded_train_sents[:train_sents.shape[0]] = train_sents
        padded_batch.append(padded_train_sents)
    return padded_batch


class LSTM(nn.Module):
    def __init__(self, input_size=300, hidden_size=256, num_layers=2, label_size=2, bidirectional=True,
                 batch_first=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
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
                                                   random_state=random_state)
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


def train(model, embedding_model, dataloader, epochs: int = 60, lr: float = 1e-3, seq_len: int = 50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        for i, (train_sents, train_labels) in enumerate(dataloader):
            # create a list of word embeddings per sentence
            train_sents_tensor = [get_sentence_tensor(embedding_model=embedding_model,
                                                      sentence=str(sent),
                                                      seq_len=seq_len) for sent in train_sents]
            # ensure all sentences in the batch have the same length, hence padding
            train_sents_tensor = pad_batch(train_sents_tensor)
            # stack sentence matrices onto each other
            train_sents_tensor = torch.stack(train_sents_tensor)

            output_labels = model(train_sents_tensor)
            train_labels = train_labels.long()
            loss = criterion(output_labels, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100:
                print(f'Epoch [{epoch}]\tStep [{i + 1}/{len(dataloader.dataset)}]\tLoss [{loss.item():.4f}]')


def main():
    lr = 1e-3
    EPOCHS = 60
    input_size = 300
    seq_len = 50
    num_layers = 2
    batch_size = 16

    print('Loading Financial Word Embeddings Model')
    fasttext_cbow_path = '../resources/FinText_FastText_CBOW/Word_Embedding_2000_2015'
    embedding_model = FastText.load(fasttext_cbow_path)  # FinText: (FastText/CBOW)

    model = LSTM(input_size=input_size, num_layers=num_layers)
    print('Loading Training & Validation Data')
    training_data = FNS2021(file='../tmp/training_corpus_20230129 16:01.csv', training=True)
    # validation_data = FNS2021(file='../tmp/training_corpus_20230129 16:01.csv', training=False)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, drop_last=True)
    # validation_dataloader = DataLoader(validation_data, batch_size=batch_size, drop_last=True)
    print('Starting LSTM training')
    train(model=model, embedding_model=embedding_model, dataloader=train_dataloader, lr=lr, epochs=EPOCHS, seq_len=seq_len)


main()
