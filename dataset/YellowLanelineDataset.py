
import cv2
import glob
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class YellowLanelineDataset(Dataset):
    '''
    斑马胶带车道线数据集，mask的像素值共两类，其中0表示背景，[128, 0, 0]红色表示车道线，不存在其他值
    将mask的0值设定为label：0，mask的[128, 0, 0]值设定为label：1
    '''

    def __init__(self, resize = (160, 320)):
        # set path
        self.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_paths = [ '/home/chenyp/dataset/laneline_1_voc_202101041129/pool', 
                            '/home/chenyp/dataset/laneline_2_voc_202101071904/pool', 
                            '/home/chenyp/dataset/laneline_3_voc_202101111451/pool', 
                            '/home/chenyp/dataset/laneline_4_voc_202101111452/pool',
                            '/home/chenyp/dataset/laneline_5_voc_202101131411/pool']
        self.images_paths = [os.path.join(data_path, 'raw') for data_path in self.data_paths]
        self.masks_paths = [os.path.join(data_path, 'mask_class') for data_path in self.data_paths]
        # set images and masks name list
        self.images_list = []
        for images_path in self.images_paths:
            images_list = os.listdir(images_path)
            images_list.sort()
            images_list_absolute_path = [os.path.join(images_path, image_name) for image_name in images_list]
            self.images_list = self.images_list + images_list_absolute_path
        #print(self.images_list)
        self.masks_list = []
        for masks_path in self.masks_paths:
            masks_list = os.listdir(masks_path)
            masks_list.sort()
            masks_list_absolute_path = [os.path.join(masks_path, mask_name) for mask_name in masks_list]
            self.masks_list = self.masks_list + masks_list_absolute_path
        #print(self.masks_list)
        assert len(self.images_list) == len(self.masks_list)
        print(len(self.images_list))
        # set transform
        self.resize = resize
        self.images_transform = transforms.Compose([transforms.Resize(size = self.resize), transforms.ColorJitter(brightness=0.3,contrast=0.1,saturation=0.1,hue=0.05), transforms.ToTensor()])
        #self.images_transform = transforms.Compose([transforms.Resize(size = self.resize), transforms.ToTensor()])
        self.masks_transform = None


    def __len__(self):
        return len(self.masks_list)

    def __getitem__(self, idx):
        # read signle image and mask
        mask_name = self.masks_list[idx]
        image_name = self.images_list[idx]
        image = Image.open(image_name)
        mask = cv2.imread(mask_name)[:,:,2]
        # resize image and mask
        if self.resize:
            mask = cv2.resize(mask, (self.resize[1], self.resize[0]))
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
    
    zebra_laneline = YellowLanelineDataset()
    train_size = int(0.9 * len(zebra_laneline))
    test_size = len(zebra_laneline) - train_size
    train_dataset, test_dataset = random_split(zebra_laneline, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
