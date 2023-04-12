import nltk
from extractor import *


def experiment_comparison(project: str = 'extractive_summarisation-data-augmentation-unified', seed: int = 42):
    trainining_data = f'../tmp/train_downsample_0.8_random_{seed}.csv'
    validation_data = f'../tmp/validation_downsample_0.8_random_{seed}.csv'
    # Define W&B hyperparameter sweep
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {
        'seed': {
            'values': [seed]
        },
        'attention_type': {
            'values': ['dot', None]
        },
        'data_augmentation': {
            'values': ['fr']
        },
        'lr': {
            'values': [1e-3]
        },
        'batch_size': {
            'values': [32]
        },
        'hidden_size': {
            'values': [64]
        },
        'downsample_rate': {
            'values': [0.8]  # [0.5, 0.75, 0.9]
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


def experiment_downsample90(project: str = 'extractive_summarisation-data-augmentation-unified', seed: int = 42):
    trainining_data = f'../tmp/train_downsample_0.9_random_{seed}.csv'
    validation_data = f'../tmp/validation_downsample_0.9_random_{seed}.csv'
    # Define W&B hyperparameter sweep
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {
        'seed': {
            'values': [seed]
        },
        'attention_type': {
            'values': ['dot', None]
        },
        'data_augmentation': {
            'values': [None]
        },
        'lr': {
            'values': [1e-3]
        },
        'batch_size': {
            'values': [32]
        },
        'hidden_size': {
            'values': [64]
        },
        'downsample_rate': {
            'values': [0.9]
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
    experiment_downsample90(seed=42)
    experiment_comparison(seed=42)


main()
