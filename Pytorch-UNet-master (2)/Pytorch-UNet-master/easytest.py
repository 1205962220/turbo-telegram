import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torchvision.transforms import ToPILImage

import unet
from NewDataset import MyDataLoader

img_path = "data/imgs/"
tag_path = "data/masks/"
data_length = 10
test_rate = 0.3
batch_size = 2
Resize = 64
def img_show(img):
    '''将img转化为PIL图像格式后展示'''
    to_pil_image = ToPILImage()
    img = to_pil_image(img)
    plt.imshow(img)

for epoch in range(2):
    data_iter = iter(train_loader)
    for i, (train_features, train_labels) in enumerate(data_iter):
        plt.figure(i)
        for j in range(batch_size):
            plt.subplot(int(f"1{batch_size}{j + 1}"))
            # print(train_features.size())
            img = train_features[j]
            label = train_labels[j]
            img_show(img)
            print(f"Label: {label}")
        plt.show()

