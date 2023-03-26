import nltk
from extractor import *


def experiment1(project: str = 'extractive_summarisation-data-augmentation-'):
    # Define W&B hyperparameter sweep
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {
        'seed': {
            'values': [42, 41]
        },
        'attention_type': {
            'values': ['dot']
        },
        'data_augmentation': {
          'values': ['fr', None]
        },
        'lr': {
            'values': [1e-3]
        },
        'batch_size': {
            'values': [32]
        },
        'hidden_size': {
            'values': [64, 256]
        },
        'downsample_rate': {
            'values': [0.5, 0.75, 0.9]
        },
        'dropout': {
            'values': [0, 0.25]
        },
        'rnn_type': {
            'values': ['gru']
        },
        'epochs': {
            'value': 60
        },
        'project': {
            'value': project
        }
    }
    sweep_config['parameters'] = parameters_dict
    wandb.login(key='d688e4b0d6cc6faf80068f7320efc3f0d135e36d')
    sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, run_experiment)


def experiment_comparison(project: str = 'extractive_summarisation-data-augmentation-unified'):
    trainining_data = '../tmp/train_downsample_0.75_random_42.csv'
    validation_data = '../tmp/validation_downsample_0.75_random_42.csv'
    # Define W&B hyperparameter sweep
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {
        'seed': {
            'values': [42]
        },
        'attention_type': {
            'values': ['dot', None]
        },
        'data_augmentation': {
          'values': ['fr', None]
        },
        'lr': {
            'values': [1e-3]
        },
        'batch_size': {
            'values': [32]
        },
        'hidden_size': {
            'values': [64, 256]
        },
        'downsample_rate': {
            'values': [0.75] # [0.5, 0.75, 0.9]
        },
        'dropout': {
            'values': [0]
        },
        'rnn_type': {
            'values': ['gru']
        },
        'epochs': {
            'value': 60
        },
        'project': {
            'value': project
        },
        'training_data': {
            'values': [f'{trainining_data}']
        },
        'validation_data': {
            'values': [f'{validation_data}']
        }
    }
    sweep_config['parameters'] = parameters_dict
    wandb.login(key='d688e4b0d6cc6faf80068f7320efc3f0d135e36d')
    sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, run_experiment)


def main():
    nltk.download('punkt')
    # root = '..'
    experiment_comparison()


main()
