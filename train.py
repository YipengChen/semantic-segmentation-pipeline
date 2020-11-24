import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import visdom

from dataset.BagDataset import BagDataset
from models.FCN import FCNs, VGGNet


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set visualization
vis = visdom.Visdom()

# set model, loss and optimizer
vgg_model = VGGNet(requires_grad=True)
model = FCNs(pretrained_net=vgg_model, n_class=2)
model = model.to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.7)

# data loader
dataset = BagDataset()
test_size = int(0.1 * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


def train(epoch_num=50):

    train_iter_loss = []
    test_iter_loss = []
    start_time = time.time()

    # start train
    for epoch in range(epoch_num):
        
        train_loss = 0
        model.train()
        for index, (image, mask) in enumerate(train_dataloader):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(image)
            output = torch.sigmoid(output)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            iter_loss = loss.item()
            train_loss += iter_loss
            output_np = output.cpu().detach().numpy()
            output_np = np.argmax(output_np, axis=1)
            mask_np = mask.cpu().detach().numpy()
            mask_np = np.argmax(mask_np, axis=1)

            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epoch, index, len(train_dataloader), iter_loss))
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                vis.images(mask_np[:, None, :, :], win='train_label', opts=dict(title='train_label'))
        
        train_iter_loss.append(train_loss)
        vis.line(train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))


        # start test 
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for index, (image, mask) in enumerate(test_dataloader):
                image, mask = image.to(device), mask.to(device)
                output = model(image)
                output = torch.sigmoid(output)
                loss = criterion(output, mask)
                
                iter_loss = loss.item()
                test_loss += iter_loss
                output_np = output.cpu().detach().numpy()
                output_np = np.argmax(output_np, axis=1)
                mask_np = mask.cpu().detach().numpy()
                mask_np = np.argmax(mask_np, axis=1)
        
                if np.mod(index, 15) == 0:
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction')) 
                    vis.images(mask_np[:, None, :, :], win='test_label', opts=dict(title='test label'))

            test_iter_loss.append(test_loss) 
            vis.line(test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss')) 


        # count time
        end_time = time.time()
        start_time = end_time
        print('epoch {}, mean train loss = {}, mean test loss = {}, time = {}'
                .format(epoch, train_loss/len(train_dataloader), test_loss/len(test_dataloader), end_time - start_time))
        
        # save model
        if np.mod(epoch, 5) == 0:
            torch.save(model, 'checkpoints/model_{}.pt'.format(epoch))
            print('saveing checkpoints/model_{}.pt'.format(epoch))


if __name__ == "__main__":
    train(epoch_num=100)
