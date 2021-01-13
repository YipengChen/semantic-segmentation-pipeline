import cv2
import numpy as np
import os

target_dir = '/home/chenyp/dataset/laneline_5_voc_202101131411/SegmentationClassPNG'
height = 360
width = 640
black_image = np.zeros([height, width, 3]).astype(np.int8)
print(black_image.shape)
for i in range(237):
    image_name = 'left{:0>3d}0.png'.format(i)
    file_name = os.path.join(target_dir, image_name)
    cv2.imwrite(file_name, black_image)