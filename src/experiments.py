import nltk
from extractor import *


def experiment1(project: str = 'FNS-biLSTM-classification-sweep'):
    # Define W&B hyperparameter sweep
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {
        'lr': {
            'values': [0.001, 0.005, 0.01]
        },
        'batch_size': {
            'values': [64, 32, 16]
        },
        'hidden_size': {
            'values': [256, 128, 64, 32]
        },
        'downsample_rate':{
            'values': [0.9, 0.5]
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
