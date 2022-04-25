import seaborn as sns
import torch

from collections import Counter
import matplotlib.pyplot as plt

'''
Collection of visualization functions
'''

def plot_nonzero_accuracy(global_param, global_acc, struct_param, struct_acc, unstruct_param, unstruct_acc,):
    plt.figure(figsize=(10,10))
    plt.plot(global_param, global_acc, color='blue', label='Global Pruning')
    plt.plot(struct_param, struct_acc, color='magenta', label='Structured Pruning')
    plt.plot(unstruct_param, unstruct_acc, color='green', label='Unstructured Pruning')
    plt.xlabel('# nonzero parameters')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.ylim([0, 100])
    plt.title(f'MobileViT Pruning')
    plt.savefig(f'visualizations/pruning_techniques_param_acc.png', dpi=300)

def plot_train_validation(count, train_plot, valid_plot, model_size, dataset_name, type):
    plt.figure(figsize=(10,10))
    plt.plot(count, train_plot, color='blue', label='Train')
    plt.plot(count, valid_plot, color='magenta', label='Validation')
    plt.xlabel('Epochs')
    if type == 'Accuracy':
        plt.ylabel('Accuracy (%)')
    else:
        plt.ylabel('Loss')
    plt.legend()
    if type == 'Accuracy':
        plt.ylim([0, 100])
    plt.title(f'{model_size.upper()} MobileViT Training/Validation {type} - {dataset_name.upper()}')
    plt.savefig(f'visualizations/{model_size}_{dataset_name.lower()}_{type.lower()}.png', dpi=300)