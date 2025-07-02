import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import *

model = MyNet().to(device)
model.load_state_dict(torch.load('lenet_mnist_model.pth'), strict=False)
model.eval()

def fgsm_attack(image, epsilon, data_grad):
    # 计算扰动
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # 确保扰动后的数据在合法范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def fgsm_test(model, device, test_loader, epsilon):
    #对抗样本
    adv_examples = []
    #准确度计数器
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True 

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] #得到初始预测值

        if init_pred.item() != target.item():
            continue
        
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1] #得到最终预测值

        if final_pred.item() == target.item():
            correct += 1
        if epsilon == 0 and len(adv_examples) < 10:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        
        elif len(adv_examples) < 10:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((target.item(), final_pred.item(), adv_ex))

    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adv_examples

if __name__ == '__main__':
    accuracies = []
    examples = []
    epsilons = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for eps in epsilons:
        acc, ex = fgsm_test(model, device, test_loader=test_loader, epsilon=eps)
        accuracies.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, marker='o')
    plt.xticks(np.arange(0, 1.2, step=0.2))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epsilon")
    plt.show()

    cnt = 0
    plt.figure(figsize=(10, 6))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            color = "green" if orig == adv else "red"
            plt.title("{} -> {}".format(orig, adv), color=color)
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()