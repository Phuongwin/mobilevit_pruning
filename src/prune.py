# Generic Python Libraries
import yaml

# Necessary file imports
from model.mobilevit import *
from transform import *
from visualize import *

# Machine Learning Libraries
import torch
import torch.nn as nn
from torch.nn.utils import prune
from torchvision import datasets

'''
Structured Pruning - Pruning a larger part of the network (channel or layer)
    - filter pruning
'''
def prune_model_ln_structured(model, proportion = 0.5):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, 'weight', proportion, n=1, dim=1)
            prune.remove(module, 'weight')
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, 'weight', proportion, n=1, dim=1)
            prune.remove(module, 'weight')
    return model

'''
Unstructured Pruning - Find less salient connections and remove them
    - weight pruning
'''
def prune_model_l1_unstructured(model, proportion = 0.5):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                prune.l1_unstructured(module, 'bias', proportion)
                prune.remove(module, 'bias')

            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')

    return model

'''
Global Unstructured Pruning - removing across the whole model
    - Global Structured wouldn't make sense as a technique
'''
def prune_model_global_unstructured(model, proportion = .5):
    parameters_to_prune = []
    for module in model.modules():
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

'''
Model Testing
'''
def test_model(model, test_loader):
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    return (100 * test_correct // test_total), test_total


if __name__ == '__main__':
    '''
    Read Configurations and Hyperparameters
    '''
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataset_name = config['dataset'].lower()
    model_size = config['model_size'].lower()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    unpruned_path = f"./saved_models/unpruned_{model_size}_{device}_weights_{dataset_name}.pth"

    '''
    Loading Test Set
    '''
    if dataset_name == 'cifar10':
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=test_transform)
    elif dataset_name == 'cifar100':
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size = 128,
                                              num_workers = 2,
                                              drop_last = True,
                                              shuffle = True
                                              )

    '''
    Model Instantiation
    '''
    if model_size == 'xs': 
        model = mobilevit_xs()
    elif model_size == 'xxs': 
        model = mobilevit_xxs()
    else:
        if model_size != 's':
            print('Model Size does not exist - Default selected')
            model_size = 's'
        model = mobilevit_s()

    model = model.to(device)
    print(f'MobileViT {model_size}: {count_parameters(model)} parameters')

    '''
    Experimentation with all three pruning techniques
    '''
    global_nz_param_plot = []
    global_acc_plot = []
    print("Begin Global Pruning")
    for i in range(5):
        model.load_state_dict(torch.load("./saved_models/unpruned_weights.pth"))
        model = prune_model_global_unstructured(model, i * 0.05)
        non_zeros = sparsity_calculation(model)
        global_nz_param_plot.append(non_zeros)

        # torch.save(model.state_dict(), f"./saved_models/test/pruned_global_weights{i*5}.pth")

        accuracy, test_total = test_model(model, test_loader)
        global_acc_plot.append(accuracy)

        print(f'Global - Accuracy of pruned network {i*5}%: {accuracy}%')

    struct_nz_param_plot = []
    struct_acc_plot = []
    print("Begin Unstructured Pruning")
    for i in range(5):
        model.load_state_dict(torch.load("./saved_models/unpruned_weights.pth"))
        model = prune_model_ln_structured(model, i * 0.05)
        non_zeros = sparsity_calculation(model)
        struct_nz_param_plot.append(non_zeros)

        # torch.save(model.state_dict(), f"./saved_models/test/pruned_struct_weights{i*5}.pth")

        accuracy, test_total = test_model(model, test_loader)
        struct_acc_plot.append(accuracy)

        print(f'Structured - Accuracy of pruned network {i*5}%: {accuracy}%')

    unstruct_nz_param_plot = []
    unstruct_acc_plot = []
    print("Begin Unstructured Pruning")
    for i in range(5):
        model.load_state_dict(torch.load("./saved_models/unpruned_weights.pth"))
        model = prune_model_l1_unstructured(model, i * 0.05)
        non_zeros = sparsity_calculation(model)
        unstruct_nz_param_plot.append(non_zeros)

        # torch.save(model.state_dict(), f"./saved_models/test/pruned_unstruct_weights{i*5}.pth")

        accuracy, test_total = test_model(model, test_loader)
        unstruct_acc_plot.append(accuracy)

        print(f'Unstructured - Accuracy of pruned network {i*5}%: {accuracy}%')

    ### Plot visualization of all three pruning techiques on a single graph
    plot_nonzero_accuracy(global_nz_param_plot, global_acc_plot,
                          struct_nz_param_plot, struct_acc_plot,
                          unstruct_nz_param_plot, unstruct_acc_plot,
                         )