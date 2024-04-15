import os
import numpy as np
import torch
import datetime

from torch import nn
from torch.autograd import Variable

import matplotlib.pyplot as plt

from config import *
from data_loader import *
from model import *

AE = Autoencoder(z_dim)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(AE.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)




checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
AE.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

test_loader = Test_DataLoader_Mnist(numbers=list(range(10)))  #修改

input = []
output = []

for img, label in test_loader:
    x = img.view(img.size(0), -1)
    x = Variable(x)

    xhat = AE(x)
    x = x.detach().numpy()
    xhat = xhat.detach().numpy()
    x = x / 2 + 0.5
    xhat = xhat / 2 + 0.5

    label = label.numpy()
    for i in range(0, x.shape[0]):
        input.append({'img': x[i], 'label': label[i]})
        output.append({'img': xhat[i], 'label': label[i]})


def unload(x):
    unloader = transforms.ToPILImage()

    imag = x.reshape(28, 28)

    pil_image = unloader(imag)
    return pil_image


# 数据可视化
n = 6  # number of test sample
plt.figure(figsize=(12, 6))
for i in range(n):
    # 输入图像显示
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(unload(input[i]['img']), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 输出图像显示
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(unload(output[i]['img']), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 差异图像计算
    diff_img = np.abs(input[i]['img'] - output[i]['img'])

    # 差异分数计算
    diff = np.sum(diff_img)

    # 差异图像与分数显示
    ax = plt.subplot(3, n, i + 1 + n * 2)
    plt.imshow(unload(diff_img))
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_xlabel('label = %d\nscore = %.2f' % (input[i]['label'], diff))

plt.savefig("./save/result_capsule_%s.png" % datetime.datetime.now().time().strftime(r"%H_%M_%S"))
plt.show()
plt.close()