import os
import numpy as np
import torch

from torch import nn

import pylab
import matplotlib.pyplot as plt

from config import *
from data_loader import *
from model import *

AE = Autoencoder_Teacher(z_dim)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(AE.parameters(),
                             lr=learning_rate_teacher,
                             weight_decay=1e-5)

cuda = torch.cuda.is_available()
if cuda:
    AE.cuda()

train_loader = Train_DataLoader_Capsule()

losses = np.zeros(num_epochs_teacher)

def loss_visualize():
    plt.figure()
    pylab.xlim(0, num_epochs_teacher)
    plt.plot(range(0, num_epochs_teacher), losses, label='loss')
    plt.legend()
    plt.savefig(os.path.join("./save/", 'loss.pdf'))
    plt.close()

for epoch in range(num_epochs_teacher):
    i = 0
    for img,_ in train_loader:

        # 将图像张量展平成一维
        # Flatten the image tensor into a one-dimensional tensor.
        if cuda:
            x = img.cuda().view(img.size(0), -1)
        else:
            x = img.view(img.size(0), -1)

        # 计算模型输出
        # Calculate the output of the model
        xhat = AE(x)

        # 计算loss，反向传播 优化参数
        # Calculate the loss, backpropagate, and optimize the parameters
        loss = mse_loss(xhat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 该部分代码用于记录loss的值，便于观察loss的变化
        # to record the values of the loss
        # making it easier to observe the changes in the loss.
        losses[epoch] = losses[epoch] * (i / (i + 1.)) + loss * (1. / (i + 1.))
        i += 1

    loss_visualize()
    print('epoch [{}/{}], loss: {:.4f}'.format(
        epoch + 1,
        num_epochs_teacher,
        loss))
    
torch.save({
    'epoch': num_epochs_teacher,
    'model_state_dict': AE.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': learning_rate_teacher,
    }, CKPT_PATH_TEACHER)