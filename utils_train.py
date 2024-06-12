import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seg_metrics
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import copy
from tqdm.autonotebook import tqdm, trange


def train_model(model, train_loader, val_loader, criterion, optimizer, sc_plt, device, num_epochs=25):    
    val_iou_history = []
    best_model_wts = copy.deepcopy(model)
    best_iou = 0

    for epoch in range(num_epochs):
        model = model.train()
        running_loss = 0.0
        scores_train = 0
        scores_val = 0
        val_epoch_iou = 0

        for i, sample in enumerate(train_loader):                
                reference_img = sample['reference'].to(device)
                test_img = sample['test'].to(device)
                labels = (sample['label']>0).squeeze(1).type(torch.LongTensor).to(device)
                #labels = (sample['label']>0).squeeze(1).float().to(device)
                optimizer.zero_grad()

                outputs = model([reference_img, test_img]).squeeze(1).to(device)
                #outputs = torch.clamp(outputs, min=0, max=1)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                dice_value_train = seg_metrics.iou_segmentation(preds.squeeze(1).type(torch.LongTensor), (labels>0).type(torch.LongTensor))
                scores_train += dice_value_train.mean().item()
            
                if i % 2 == 0:
                    print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                        epoch+1,
                        num_epochs,
                        i * len(labels),
                        len(train_loader.dataset),
                        100. * i / len(train_loader),
                        loss.cpu().data.item()),
                        end='')

                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
                
        #validation
        model = model.eval()
        dice_value_val = 0
        with torch.no_grad():
                val_running_loss = 0.0
                for sample in val_loader:
                    reference_img_val = sample['reference'].to(device)
                    test_img_val = sample['test'].to(device)
                    #labels_val = (sample['label']>0).squeeze(1).float().to(device)
                    labels_val = (sample['label']>0).squeeze(1).type(torch.LongTensor).to(device)

                    outputs_val = model([reference_img_val, test_img_val]).squeeze(1).to(device)
                    loss_val = criterion(outputs_val, labels_val)
                    val_running_loss += loss_val.item()
                    _, preds_val = torch.max(outputs_val, 1)

                    dice_value_val = seg_metrics.iou_segmentation(preds_val.squeeze(1).type(torch.LongTensor), (labels_val>0).type(torch.LongTensor))
                    scores_val += dice_value_val.mean().item()

        val_epoch_iou = scores_val / len(val_loader)
        train_epoch_iou = scores_train / len(train_loader)
        print()
        print(f'train loss={running_loss / len(train_loader)}')
        print(f'mean train iou={train_epoch_iou}')
        print(f'val los={val_running_loss / len(val_loader)}')
        print(f'mean val iou={val_epoch_iou}')
        val_iou_history.append(val_epoch_iou)

                
        if val_epoch_iou > best_iou:
                best_iou = val_epoch_iou
                best_model_wts = copy.deepcopy(model) 

        sc_plt.step(running_loss / len(train_loader))  
        
    return best_model_wts, val_iou_history
            
            
                
                

            
                
    
    