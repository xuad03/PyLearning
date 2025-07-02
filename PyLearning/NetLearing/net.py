import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
import random


# Generate the real data
feature = np.linspace(2, 100, 1000)
label = np.log(feature) + np.random.rand(1000) * 0.2
feature = np.array(feature, np.float32)
label = np.array(label, np.float32)
"""
plt.figure(figsize=(10, 6))
plt.plot(feature, label)
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Real Data Visualization')
plt.show()
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))

# Define the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(1, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1) 
        )

    def forward(self, x):
        logits = self.linear_relu(x)
        return logits

data_loader = DataLoader(np.concatenate((feature.reshape(1000, -1), label.reshape(1000, -1)), axis=1), \
                            batch_size=10)
batch_size = 10
# learning rate, Momentum
lr = 1e-5
momentum = 0.99

# Define the training function
def train(dataloader, model, loss_fn, optimizer):
    loss = 0
    for batch, data in enumerate(data_loader):
        x = data[:,0].reshape(10, -1).to(device)
        y = data[:,1].reshape(10, -1).to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x)
            print("loss: {0}, current samples number: {1}".format(loss, current))
        
    return loss
    
if __name__ == '__main__':
    model = Net().to(device)
    print(model)

    # Define optimizer and loss function
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    epochs = 500
    loss_list = []

    for epoch in range(epochs):
        loss_val = train(data_loader, model, loss, optimizer)
        if epoch % 10 == 0:
            # ！！！注意：这里的pred和label都是numpy数组，需要用.data.cpu()
            loss_list.append(loss_val.data.cpu().numpy())
        if epoch % 100 == 0:
            print("Epoch {0}, loss: {1}".format(epoch, loss_val))
    print("over!")

    # Plot the loss
    plt.plot([e for e in range(0, epochs, batch_size)], loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss of MSE')
    plt.title('Training Loss, lr = {}, momentum = {}'.format(lr, momentum))
    plt.show()

    torch.save(model.state_dict(), 'model_{},{}.pth'.format(lr, momentum))
    print("Model saved to model.pth")