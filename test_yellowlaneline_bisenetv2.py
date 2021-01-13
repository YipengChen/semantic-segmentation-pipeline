import cv2
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
import time


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set model
model_1 = torch.load('./checkpoints/model_950_202101111659.pt')
model_2 = torch.load('./checkpoints/model_150.pt')
model_1.eval()
model_1 = model_1.to(device)
model_2.eval()
model_2 = model_2.to(device)
#torch.save(model_2.state_dict(), 'checkpoints/yellow_laneline_seg_model_e950_202101111719.pth')
image_transforms = transforms.Compose([transforms.ToTensor()])

test_image_path = '/home/chenyp/dataset/laneline_1'
images_list = os.listdir(test_image_path)
images_list.sort()
with torch.no_grad():
    for image_name in images_list:
        images_absulote_path = os.path.join(test_image_path, image_name)
        image = cv2.imread(images_absulote_path)
        print(images_absulote_path)
        time_1 = time.time()
        image = cv2.resize(image, (320, 160))
        image_t=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_t = image_transforms(image_t)
        image_t = image_t[np.newaxis, :, :, :]
        image_t = image_t.to(device)
        
        output_1, *output_aux = model_1(image_t)
        output_1 = output_1.cpu().numpy()
        output_1 = output_1[0]
        print(output_1.shape)
        output_1 = output_1.argmax(0)
        print(time.time() - time_1)
        print(output_1.shape)
        image_1 = image.copy()
        image_1[np.where(output_1 == 1)] = [255, 0, 0]
        cv2.imshow('mask_1', image_1)

        output_2, *output_aux = model_2(image_t)
        output_2 = output_2.cpu().numpy()
        output_2 = output_2[0]
        print(output_2.shape)
        output_2 = output_2.argmax(0)
        print(output_2.shape)
        image_2 = image.copy()
        image_2[np.where(output_2 == 1)] = [255, 0, 0]
        cv2.imshow('mask_2', image_2)
        cv2.waitKey(0)