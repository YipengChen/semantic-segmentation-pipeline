
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BagDataset(Dataset):
    '''
    背包数据集，共600对图像数据，图像大小不等，mask的像素指共两类，其中0表示背包，255表示背景，存在0～255中间值
    将mask的128～255值设定为label：0，mask的0～127值设定为label：1

    '''

    def __init__(self, resize = (160, 160)):
        # set path
        self.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.project_path, 'data', 'bag')
        self.images_path = os.path.join(self.data_path, 'images')
        self.masks_path = os.path.join(self.data_path, 'masks')
        # set images and masks name list
        self.images_list = os.listdir(self.images_path)
        self.masks_list = os.listdir(self.masks_path)
        assert len(self.images_list) == len(self.masks_list)
        # set transform
        self.images_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.masks_transform = None
        # set parameters
        self.resize = resize
        if self.resize:
            assert len(self.resize) == 2

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # read signle image and mask
        image_name = self.images_list[idx]
        image = cv2.imread(os.path.join(self.images_path, image_name))
        mask = cv2.imread(os.path.join(self.masks_path, image_name), 0)
        # resize image and mask
        if self.resize:
            image = cv2.resize(image, self.resize)
            mask = cv2.resize(mask, self.resize)
        # image process       
        if self.images_transform:
            image = self.images_transform(image)    
        # mask process
        mask = self.maks_preprocess(mask)
        if self.masks_transform:
            mask = self.masks_transform(mask)
        # return
        return image, mask

    def maks_preprocess(self, mask):
        mask = (mask/255).astype('uint8')
        buf = np.zeros((2, ) + mask.shape)
        buf[0, mask==1], buf[1, mask==0] = 1, 1
        assert np.sum(buf, axis = 0).all() == 1
        buf = torch.tensor(buf, dtype = torch.float)
        return buf


if __name__ =='__main__':

    from torch.utils.data import DataLoader, random_split
    
    bag = BagDataset()
    train_size = int(0.9 * len(bag))
    test_size = len(bag) - train_size
    train_dataset, test_dataset = random_split(bag, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
