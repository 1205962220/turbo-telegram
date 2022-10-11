import os

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def read_directory(directory_name, array_of_img, length, imgPath):
    # 参数为文件夹名称，图像数组，读取的文件长度，图像地址数组
    direct = os.listdir(r"./" + directory_name)
    # 乱序读取进来的数据集，对读取进来的数据集按照标号进行排序
    direct.sort(key=lambda x: int(x[:-4]))
    # 如果输入-1，将其转换为长度加一，即将数据全部读入
    if length == -1:
        length = len(direct) + 1
    for filename in direct[0: length]:
        print("读取图像：" + filename)  # 测试是否按序读入
        iPath = "./" + directory_name + filename  # 读取当前图像的地址
        imgPath.append(iPath)  # 追加进图像地址数组中去
        img = cv2.imread(directory_name + "/" + filename)
        # 调整图像大小
        # img = cv2.resize(img, (32, 32))
        # 使用cv2.imread()接口读图像，读进来的是BGR格式以及【0～255】,所以需要将img转换为RGB格式正常显示
        img = img[:, :, [2, 1, 0]]
        # # 转为一维数组
        # img = np.array(img).flatten()
        # # print(img)
        array_of_img.append(img)




#显示图像
def show_pic(images, length):
    i = 0
    plt.figure(figsize=(length, length))
    for img in images[0: length * length]:
        plt.subplot(length, length, i + 1)
        plt.imshow(img)
        # 关闭x，y轴显示
        plt.xticks([])
        plt.yticks([])
        i += 1

    plt.show()

#读取标签图片
def read_tag(tag_path, tags, length):
    # 参数为标签文件路径，标签数组，读取长度
    # 读取数据的标签
    with open(tag_path) as file:
        for line in list(file)[0:length]:
            # 读取前多少张图片的标签
            a = str(line).split("\t")
            b = a[1]  # 得到数字值
            tag = int(b[0])  # 去掉换行符
            tags.append(tag)


#  对数据进行处理与加载
def data_load(img_path, tag_path, length, test_rate):
    # 获取数据与数据标签，并将数据分为训练集和测试集
    # 参数为图像文件夹，标签地址，length为读取的数据集长度，size为训练集测试集的区分比例

    images = []  # 图像数组
    tags = []  # 标签数组
    imPath = []  # 图像地址数组
    read_directory(img_path, images, length, imPath)


    # 测试图像读取是否正常
    # show_pic(images, 10)

    # 转换为numpy数组
    images = np.array(images)
    tags = np.array(tags)

    # 返回处理得到的数据，有 X_train, X_test, y_train, y_test 四部分
    return train_test_split(images, tags, test_size=test_rate)

def create_txt(img_path, tag_path, length, test_rate):
    # 生成训练集，验证集的数据地址与对应标签存储的数据对
    images = []  # 图像数组
    imagePath = []  # 图像地址数组
    tags = []  # 训练标签数组
    tagsPath = []

    # 读取指定数量的图像，图像地址，以及对应的训练标签
    read_directory(img_path, images, length, imagePath)
    read_directory(tag_path, tags, length, tagsPath)

    # 根据比例随机划分训练集与测试集对应的图像地址和对应的标签
    X_train, X_test, y_train, y_test = train_test_split(imagePath, tagsPath, test_size=test_rate)

    # w+意思是：
    # 打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。
    # 如果该文件不存在，创建新文件
    train = open('./data/train.txt', 'w+')
    test = open('./data/test.txt', 'w+')

    for i in range(len(X_train)):  # 生成训练文档
        str(y_train[i])
        X_train[i]
        name = X_train[i] + ' ' + '\n'
        train.write(name)
    for i in range(len(X_test)):  # 生成测试文档
        name = X_test[i] + ' ' + '\n'
        test.write(name)

    train.close()
    test.close()
img_path = "data/imgs/"
tag_path = "data/masks/"
create_txt(img_path,tag_path,5,0.5)
#
# def default_loader(path):
#     return Image.open(path).convert('RGB')
#
#
# class MyDataset(Dataset):
#     # 使用__init__()初始化一些需要传入的参数及数据集的调用
#     def __init__(self, txt, resize, target_transform=None, loader=default_loader):
#         super(MyDataset, self).__init__()
#         # 对继承自父类的属性进行初始化
#         fh = open(txt, 'r')
#         imgs = []
#         # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
#         for line in fh:  # 迭代该列表,按行循环txt文本中的内容
#             line = line.strip('\n')
#             line = line.rstrip('\n')
#             # 删除本行string字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符
#             words = line.split()
#             # 用split将该行分割成列表，split的默认参数是空格，所以不传递任何参数时分割空格
#             imgs.append((words[0], int(words[1])))
#             # 把txt里的内容读入imgs列表保存
#         self.imgs = imgs
#         # 重新定义图像大小
#         self.transform = transforms.Compose([transforms.Resize(size=(resize, resize)), transforms.ToTensor()])
#         self.target_transform = target_transform
#         self.loader = loader
#
#     # 使用__getitem__()对数据进行预处理并返回想要的信息
#     def __getitem__(self, index):
#         fn, label = self.imgs[index]
#         # fn是图片path
#         img = self.loader(fn)
#         # 按照路径读取图片
#         if self.transform is not None:
#             img = self.transform(img)
#             # 数据标签转换为Tensor
#         return img, label
#         # return回哪些内容，那么在训练时循环读取每个batch时，就能获得哪些内容
#
#     # 使用__len__()初始化一些需要传入的参数及数据集的调用
#     def __len__(self):
#         return len(self.imgs)
#
# def MyDataLoader(img_path, tag_path, length, test_rate, Resize, batch_size):
#     # 制作适用于torch的数据
#     create_txt(img_path, tag_path, length, test_rate)
#     train_data = MyDataset(txt='./data/train.txt', resize=Resize)
#     test_data = MyDataset(txt='./data/test.txt', resize=Resize)
#     train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader
#
# def main():
#     img_path = "data/imgs"
#     tag_path = "data/masks"
#     data_length = 10
#     test_rate = 0.3
#     batch_size = 2
#     Resize = 64
#
#     train_loader, test_loader = MyDataLoader(img_path, tag_path, data_length, test_rate, Resize, batch_size)
#     for X, y in train_loader:
#         print("Shape of X [N, C, H, W]: ", X.shape)
#         print("Shape of y: ", y.shape, y.dtype)
#         break
#
#
# if __name__ == "__main__":
#     main()
