import os
import numpy as np
import torch

from torch import nn
from torch.autograd import Variable

import pylab
import matplotlib.pyplot as plt

from config import *
from data_loader import *
from model import *

AE = Autoencoder(z_dim)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(AE.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)

if cuda:
    AE.cuda()

train_loader = Train_DataLoader_Capsule()

losses = np.zeros(num_epochs)

for epoch in range(num_epochs):
    i = 0
    for img,_ in train_loader:

        x = img.view(img.size(0), -1)

        if cuda:
            x = Variable(x).cuda()
        else:
            x = Variable(x)

        xhat = AE(x)

        #算loss
        loss = mse_loss(xhat, x)
        losses[epoch] = losses[epoch] * (i / (i + 1.)) + loss * (1. / (i + 1.))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1

    plt.figure()
    pylab.xlim(0, num_epochs)
    plt.plot(range(0, num_epochs), losses, label='loss')
    plt.legend()
    plt.savefig(os.path.join("./save/", 'loss.pdf'))
    plt.close()

    print('epoch [{}/{}], loss: {:.4f}'.format(
        epoch + 1,
        num_epochs,
        loss))
    
torch.save({
    'epoch': num_epochs,
    'model_state_dict': AE.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': learning_rate,
    }, CKPT_PATH)