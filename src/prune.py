from main import mobilevit_xxs
from model.mobilevit import *
from transform import *

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
        # print("Zeros", int(zeros))
        # print("Non-zeros", int(non_zeros + zeros))

    print(f"Non-Zero Parameters: {total_nonzeros}")
    print(f"Total Parameters: {total_param}")
    print(f"Sparsity: {float(total_zeros/total_param)}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = mobilevit_xxs()
    model = model.to(device)
    print(count_parameters(model))

    model.load_state_dict(torch.load("./saved_models/unpruned_weights.pth"))

    new_model = copy.deepcopy(model)

    sparsity_calculation(new_model)

    new_model = prune_model_l1_unstructured(new_model)

    torch.save(model.state_dict(), "./saved_models/pruned_weights.pth")

    print("Testing Pruned")

    cifar_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(cifar_test,
                                              batch_size = 128,
                                              num_workers = 2,
                                              drop_last = True,
                                              shuffle = True
                                              )

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = new_model(images)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {test_total} test images: {100 * test_correct // test_total} %')