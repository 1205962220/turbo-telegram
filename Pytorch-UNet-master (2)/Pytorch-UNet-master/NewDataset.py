from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from Dataloader import create_txt


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # 使用__init__()初始化一些需要传入的参数及数据集的调用
    def __init__(self, txt, resize, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        imgs = []
        label = []
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表,按行循环txt文本中的内容
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除本行string字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符
            words = line.split()
            # 用split将该行分割成列表，split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0]))
            # label.append(( words[1]))
            # 把txt里的内容读入imgs列表保存
        self.imgs = imgs
        self.labels = label
        # 重新定义图像大小
        self.transform = transforms.Compose([transforms.Resize(size=(resize, resize)), transforms.ToTensor()])
        self.target_transform = target_transform
        self.loader = loader

    # 使用__getitem__()对数据进行预处理并返回想要的信息
    def __getitem__(self, index):
        
        fn = self.imgs[index]
        # label = self.labels[index]
        # fn是图片path
        img = self.loader(fn)
        # lable = self.loader(label)

        if self.transform is not None:
            img = self.transform(img)
            # img = img.unsqueeze(0)
            # lable = self.transform(label)

            # 数据标签转换为Tensor
        return img
        # return回哪些内容，那么在训练时循环读取每个batch时，就能获得哪些内容

    # 使用__len__()初始化一些需要传入的参数及数据集的调用
    def __len__(self):
        return len(self.imgs)

def MyDataLoader(img_path, tag_path, length, test_rate, Resize, batch_size):
    # 制作适用于torch的数据
    create_txt(img_path, tag_path, length, test_rate)
    train_data = MyDataset(txt='./data/train.txt', resize=Resize)
    test_data = MyDataset(txt='./data/test.txt', resize=Resize)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    img_path = "data/imgs/"
    tag_path = "data/masks/"
    data_length = 10
    test_rate = 0.1
    batch_size = 2
    Resize = 64

    train_loader, test_loader = MyDataLoader(img_path, tag_path, data_length, test_rate, Resize, batch_size)
    for X, y in train_loader:
        print("Shape of X [N, C, H, W]: ", X.shape,X.type())
        print("Shape of y: ", y.shape, y.shape)
        break


if __name__ == "__main__":
    main()