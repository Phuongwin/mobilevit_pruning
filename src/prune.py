from main import mobilevit_xxs
from model.mobilevit import *
from transform import *
from visualize import *

import torch
import torch.nn as nn
from torch.nn.utils import prune
from torchvision import datasets

import copy

def prune_model_ln_structured(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, 'weight', 0.1, n=1, dim=1)
            prune.remove(module, 'weight')
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, 'weight', 0.1, n=1, dim=1)
            prune.remove(module, 'weight')
    return model

def prune_model_l1_unstructured(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, 'weight', 0.3)
            prune.remove(module, 'weight')
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                print("bias pruned")
                print(module.bias)
                prune.l1_unstructured(module, 'bias', 0.3)
                prune.remove(module, 'bias')
                print(module.bias)

            print("weight pruned")
            prune.l1_unstructured(module, 'weight', 0.3)
            prune.remove(module, 'weight')
        else:
            print("No Pruning")

    return model

def prune_model_global_unstructured(model, proportion=.6):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
        elif isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(parameters_to_prune, 
                              pruning_method = prune.L1Unstructured,
                              amount = proportion)

    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    return model

'''
Sparsity = 1 - (count_nonzero(model)/total_elements(model))
'''
def sparsity_calculation(model):
    total_zeros = 0
    total_nonzeros = 0
    total_param = 0
    for i in list(model.state_dict().values()):
        zeros = torch.sum(i == 0)
        non_zeros = torch.count_nonzero(i)
        total_zeros += int(zeros)
        total_nonzeros += int(non_zeros)
        total_param += int(non_zeros + zeros)

    sparsity = round(1 - float(total_nonzeros/total_param), 4)
    print(f"Non-Zero Parameters: {total_nonzeros}")
    print(f"Total Parameters: {total_param}")
    print(f"Sparsity: {sparsity}")
    
    return total_nonzeros

def test_model(model,test_loader):
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            # ouputs_original = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    return (100 * test_correct // test_total), test_total

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    '''
    Loading Test Set
    '''
    cifar_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(cifar_test,
                                              batch_size = 128,
                                              num_workers = 2,
                                              drop_last = True,
                                              shuffle = True
                                              )

    '''
    Model Instantiation
    '''
    model = mobilevit_xxs()
    model = model.to(device)
    print(count_parameters(model))

    model.load_state_dict(torch.load("./saved_models/unpruned_weights.pth"))

    new_model = copy.deepcopy(model)
    nz_param_plot = []
    acc_plot = []

    print("Begin Global Unstructured Pruning")
    for i in range(20):
        new_model.load_state_dict(torch.load("./saved_models/unpruned_weights.pth"))
        new_model.to(device)
        new_model = prune_model_global_unstructured(new_model, i * 0.05)
        non_zeros = sparsity_calculation(new_model)
        nz_param_plot.append(non_zeros)

        #torch.save(new_model.state_dict(), f"./saved_models/test/pruned_weights{i*5}.pth")

        accuracy, test_total = test_model(new_model, test_loader)
        acc_plot.append(accuracy)

        print(f'Accuracy of pruned network {i*5}%: {accuracy}%')

    plot_nonzero_accuracy(nz_param_plot, acc_plot)