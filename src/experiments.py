import nltk
from extractor import *


def experiment1(project: str = 'FNS-biLSTM-classification-sweep'):
    # Define W&B hyperparameter sweep
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {
        'lr': {
            'values': [0.01, 0.005, 0.001]
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'hidden_size': {
            'values': [32, 64, 128, 256]
        },
        'epochs': {
            'value': 60
        },
        'project': {
            'value': project
        }
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, run_experiment)


def main():
    nltk.download('punkt')
    # root = '..'
    experiment1()


main()
