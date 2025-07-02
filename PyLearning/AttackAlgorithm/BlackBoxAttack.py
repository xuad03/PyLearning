import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import *
from ResNet18 import *
from FGSM import *
from PGD import *

ResNet18_model = ResNet18(num_classes=10).to(device)
ResNet18_model.load_state_dict(torch.load('ResNet18_model.pth'))
ResNet18_model.eval()

MyModel = MyNet().to(device)
MyModel.load_state_dict(torch.load('lenet_mnist_model.pth'))
MyModel.eval()

def BlackBoxAttack_ProxyModel_PGD(proxy_model, attacked_model, test_loader, device):
    """
    This function is used to perform black-box attack on the attacked_model using the proxy_model as a black-box model.
    """
    correct = 0
    examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        adv_data = PGD_Attack(proxy_model, data, target, epsilon_pgd, alpha_pgd, num_iter_pgd)
        logits, output = attacked_model(adv_data)
        pred = output.argmax(dim=1, keepdim=True)
        
        if pred.item() == target.item():
            correct += 1
        if len(examples) < 10:
            output_data = data.squeeze().detach().cpu().numpy()
            examples.append((target.item(), pred.item(), output_data))
            
    final_acc = correct / len(test_loader)
    print("Test Accuracy = {} / {} = {}".format(correct, len(test_loader), final_acc))
    return examples

if __name__ == '__main__':
    epsilon_pgd = 0.50
    num_iter_pgd = 10
    alpha_pgd = 1.0

    ex_list = BlackBoxAttack_ProxyModel_PGD(MyModel, ResNet18_model,test_loader, device)
    
    plt.figure(figsize=(10, 6))
    cnt = 0
    for j in range(len(ex_list)):
        cnt += 1
        plt.subplot(1, len(ex_list), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        orig, adv, ex = ex_list[j]
        color = "green" if orig == adv else "red"
        plt.title("{} -> {}".format(orig, adv), color=color, fontsize=18)
        plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()