import seaborn as sns
import torch

from collections import Counter
import matplotlib.pyplot as plt

'''
Collection of visualization functions
'''

def plot_nonzero_accuracy(count, mobilevit_acc):
    plt.figure(figsize=(10,10))
    plt.plot(count, mobilevit_acc, color='blue', label='Global Structured Pruning')
    plt.xlabel('# nonzero parameters')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.ylim([0, 100])
    plt.title(f'MobileViT Sparsity')
    plt.savefig(f'visualizations/nonzero_accuracy.png', dpi=300)

def plot_train_validation(count, train_plot, valid_plot, experiment, type):
    plt.figure(figsize=(10,10))
    plt.plot(count, train_plot, color='blue', label='Train')
    plt.plot(count, valid_plot, color='magenta', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.title(f'{experiment} CNN Training/Validation {type}')
    plt.savefig(f'visualizations/{experiment.lower()}_{type.lower()}.png', dpi=300)

if __name__ == "__main__":
    nz_count = [1019299, 969150, 919001, 868853, 818704, 768555, 718406, 668257, 618109, 567960]

    acc = [60, 61, 60, 61, 60, 61, 60, 60, 60, 60]

    plot_nonzero_accuracy(nz_count, acc)