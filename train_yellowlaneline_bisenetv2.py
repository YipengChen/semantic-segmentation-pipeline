import numpy as np
import time
import torch
from torch.utils.data import DataLoader, random_split
import visdom

from dataset.YellowLanelineDataset import YellowLanelineDataset
from loss.ohem_ce_loss import OhemCELoss
from models.bisenetv2 import BiSeNetV2


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set visualization
vis = visdom.Visdom()

# set model, loss and optimizer
model = BiSeNetV2(n_classes = 2)
model = model.to(device)
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# data loader
dataset = YellowLanelineDataset()
test_size = int(0.1 * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)


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
            output, *output_aux = model(image)
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
                vis.images(image,win='raw_image', opts=dict(title='train raw_image'))
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                vis.images(mask_np[:, None, :, :], win='train_label', opts=dict(title='train_label'))
        
        train_iter_loss.append(train_loss)
        vis.line(train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))


        # start test 
        test_loss = 0
        model.eval()
        test_start_time = time.time()
        with torch.no_grad():
            for index, (image, mask) in enumerate(test_dataloader):
                image, mask = image.to(device), mask.to(device)
                output, *output_aux = model(image)
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
        print('epoch {}, mean train loss = {}, mean test loss = {}, sum time = {}'
                .format(epoch, train_loss/len(train_dataloader), test_loss/len(test_dataloader), end_time - start_time))
        start_time = end_time

        # save model
        if np.mod(epoch, 50) == 0:
            torch.save(model, 'checkpoints/model_{}.pt'.format(epoch))
            print('saveing checkpoints/model_{}.pt'.format(epoch))


if __name__ == "__main__":
    train(epoch_num=1000)
