import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
from net import Net, train
from net import feature, label, data_loader, batch_size, device

def model_tuning(parameters_str, lr, momentum, epochs = 100):
    plt.figure(figsize=(10, 6))
    tuning_parameters = lr if parameters_str == 'lr' else momentum
    loss_min = []
    for i in range(len(tuning_parameters)):
        model = Net().to(device)
        optimizer = None
        if parameters_str == 'lr':
            optimizer = torch.optim.SGD(model.parameters(), lr=tuning_parameters[i], momentum=1)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=tuning_parameters[i])
        loss_plot = []
        for e in range(epochs):
            loss_val = train(data_loader, model, loss_fn=nn.MSELoss(), optimizer = optimizer)
            if e % 10 == 0:
                loss_plot.append(loss_val.data.cpu().numpy())
        
        print("================================================================\n \
              when {0} = {1}, min_loss = {2}"     \
              "\n================================================================".format(parameters_str,tuning_parameters[i], np.min(loss_plot)))
        loss_min.append(np.min(loss_plot))
        plt.plot([e for e in range(0, epochs, batch_size)], loss_plot, label=tuning_parameters[i])
    print("The best {} is {}, the los is {}       \
          ".format(parameters_str, tuning_parameters[np.argmin(loss_min)], np.min(loss_min)))

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss of MSE")
    plt.title("Model Tuning - {}".format(parameters_str))
    plt.show()

if __name__ == '__main__':
    # lr = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    momentum = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    #model_tuning('lr', lr, 0.9)
    model_tuning('momentum', 0.00001, momentum)