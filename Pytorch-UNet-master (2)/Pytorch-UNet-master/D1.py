import os
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from torch import optim
from torch.utils import data
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

import unet

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1,1]
])


# 定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self, root):
        # 所有图片的绝对路径
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    batch_size = 2
    dataSet = FlameSet('./data/imgs')
    train_loader = DataLoader(dataset=dataSet, batch_size=4, shuffle=True)


    def img_show(img):
        '''将img转化为PIL图像格式后展示'''
        to_pil_image = ToPILImage()
        img = to_pil_image(img)
        plt.imshow(img)


net = unet.UNet(3, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels= data
        #  =
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(i, running_loss)
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

for epoch in range(2):
    data_iter = iter(train_loader)
    for i, train_features in enumerate(data_iter):
        plt.figure(i)
        for j in range(batch_size):
            plt.subplot(int(f"1{batch_size}{j + 1}"))
            # print(train_features.size())
            img = train_features[j]
            # label = train_labels[j]
            img_show(img)
            print(f"Label: {img.size(),i}")
        plt.show()

print(dataSet[0].shape,len(dataSet))