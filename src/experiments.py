import nltk
from extractor import *


def experiment1(project: str = 'extractive_summarisation'):
    # Define W&B hyperparameter sweep
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {
        'lr': {
            'values': [1e-3]
        },
        'batch_size': {
            'values': [32]
        },
        'hidden_size': {
            'values': [256]
        },
        'downsample_rate': {
            'values': [0.9]
        },
        'dropout': {
          'values': [0, 0.25, 0.5]
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
