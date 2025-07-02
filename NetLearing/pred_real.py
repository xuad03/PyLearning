import torch
import matplotlib.pyplot as plt
import numpy as np
from net import Net
from net import feature, label, lr


def pred_real():
    device = torch.device('cpu') #numpy cpu only
    model = Net().to(device)
    model.load_state_dict(torch.load('model_1e-05,0.99.pth'))

    pred = model(torch.tensor(feature).reshape(-1, 1))
    pred = pred.reshape(1000).cpu().detach().numpy()

    plt.plot(feature, label, color='blue', label='real data')
    plt.plot(feature, pred, color='red', label='predicted data', marker='o')
    plt.xlabel('Feature')
    plt.ylabel('Label')
    plt.title('Predicted vs Real Data, lr = {}'.format(lr))
    plt.legend()

    plt.show()

if __name__ == '__main__':
    pred_real()