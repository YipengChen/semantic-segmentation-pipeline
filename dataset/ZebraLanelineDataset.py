
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ZebraLanelineDataset(Dataset):
    '''
    斑马胶带车道线数据集，mask的像素值共两类，其中0表示背景，[128, 0, 0]红色表示车道线，不存在其他值
    将mask的0值设定为label：0，mask的[128, 0, 0]值设定为label：1
    '''

    def __init__(self, resize = (320, 160)):
        # set path
        self.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = '/home/chenyp/dataset/zebra_laneline/pool'
        self.images_path = os.path.join(self.data_path, 'raw')
        self.masks_path = os.path.join(self.data_path, 'mask_class')
        # set images and masks name list
        self.images_list = os.listdir(self.images_path)
        self.masks_list = os.listdir(self.masks_path)
        #assert len(self.images_list) == len(self.masks_list)
        # set transform
        self.images_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.masks_transform = None
        # set parameters
        self.resize = resize
        if self.resize:
            assert len(self.resize) == 2

    def __len__(self):
        return len(self.masks_list)

    def __getitem__(self, idx):
        # read signle image and mask
        mask_name = self.masks_list[idx]
        image_name = mask_name.split('.')[0] + '.jpg'
        image = cv2.imread(os.path.join(self.images_path, image_name))
        mask = cv2.imread(os.path.join(self.masks_path, mask_name))[:,:,2]
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
        mask = (mask/128).astype('uint8')
        buf = np.zeros((2, ) + mask.shape)
        buf[0, mask==0], buf[1, mask==1] = 1, 1
        assert np.sum(buf, axis = 0).all() == 1
        buf = torch.tensor(buf, dtype = torch.float)
        return buf


if __name__ =='__main__':

    from torch.utils.data import DataLoader, random_split
    
    zebra_laneline = ZebraLanelineDataset()
    train_size = int(0.9 * len(zebra_laneline))
    test_size = len(zebra_laneline) - train_size
    train_dataset, test_dataset = random_split(zebra_laneline, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
