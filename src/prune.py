import torch
import torch.nn as nn
from torch.nn.utils import prune

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
