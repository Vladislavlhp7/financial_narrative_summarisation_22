import nltk
from torch.utils.data import DataLoader

from extractor import *
from src.query import get_embedding_model


def experiment1(root: str = '..'):
    lr = 1e-3
    EPOCHS = 16
    batch_size = 16
    hidden_size = 256

    # Load Embeddings directly from FastText model
    embedding_model = get_embedding_model(root=root)

    print('Loading Training & Validation Data')
    data_filename = 'training_corpus_2023-02-07 16-33.csv'
    training_data = FNS2021(file=f'{root}/tmp/{data_filename}', training=True)
    validation_data = FNS2021(file=f'{root}/tmp/{data_filename}', training=False)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, drop_last=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, drop_last=True)

    run_training_experiment(lr=lr, epochs=EPOCHS, batch_size=batch_size, hidden_size=hidden_size,
                            embedding_model=embedding_model, train_dataloader=train_dataloader,
                            validation_dataloader=validation_dataloader)


def main():
    nltk.download('punkt')
    root = '..'
    experiment1(root=root)


main()
