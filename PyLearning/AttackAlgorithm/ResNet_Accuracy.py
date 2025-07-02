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

ResNet18_model = ResNet18(num_classes=10).to(device)
ResNet18_model.load_state_dict(torch.load('ResNet18_model.pth'))
ResNet18_model.eval()

def ResNet18_Accuracy(model, device, test_loader):
    correct = 0
    examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits, output = ResNet18_model(data)
        final_pred = output.argmax(dim=1, keepdim=True)

        if final_pred.item() == target.item():
            correct += 1
            
        if len(examples) < 10:
            output_data = data.squeeze().detach().cpu().numpy()
            examples.append((target.item(), final_pred.item(), output_data))

    final_acc = correct / len(test_loader)
    print("Test Accuracy = {} / {} = {}".format(correct, len(test_loader), final_acc))
    return examples

if __name__ == '__main__':
    ex_list = ResNet18_Accuracy(ResNet18_model, device, test_loader)
    
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