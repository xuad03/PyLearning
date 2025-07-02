import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import *

def PGD_Attack(model, image, target, epsilon, alpha, num_iter):
    perturbed_image = image.clone().detach()
    
    for i in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        
        adv_image_update = perturbed_image + alpha * perturbed_image.grad.sign()
        eta = torch.clamp(adv_image_update - image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(image + eta, min=0, max=1).detach()
        
    return perturbed_image

def PGD_test(model, test_loader, epsilon, alpha, num_iter):
    correct = 0
    adv_examples_pgd = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output_orig = model(data)
        init_pred = output_orig.max(1, keepdim=True)[1]
        
        perturbed_data = PGD_Attack(model, data, target, epsilon, alpha, num_iter)
        output_adv = model(perturbed_data)
        final_pred = output_adv.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
        if final_pred.item() != target.item() and len(adv_examples_pgd) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples_pgd.append((target.item(), final_pred.item(), adv_ex))

    final_acc = correct/float(len(test_loader))
    print('Epsilon: {:.4f},\n Alpha: {:.4f},\n Iteration: {:d},\n Test Accuracy: {:.4f}'.format(epsilon, alpha, num_iter, final_acc))
    return final_acc, adv_examples_pgd

#超参数
epsilon_pgd = 0.50
num_iter_pgd = 10
alpha_pgd = 1.0

model = MyNet().to(device)
model.load_state_dict(torch.load('lenet_mnist_model.pth'), strict=False)
model.eval()

if __name__ == '__main__':
    acc, ex_list = PGD_test(model, test_loader, epsilon_pgd, alpha_pgd, num_iter_pgd)
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