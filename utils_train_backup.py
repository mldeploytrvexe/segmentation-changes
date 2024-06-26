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


def train_model(model, dataloaders, criterion, optimizer, sc_plt, writer, device, num_epochs=25):    
    val_acc_history = []
    val_iou_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    iterations = 0
    best_iou = 0
    epoch_iou = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)        
        # Each epoch has a training and validation phase
        for phase in tqdm(['train', 'val']):            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            list_dice_val = []

            # Iterate over data.
            for sample in dataloaders[phase]:                
                reference_img = sample['reference'].to(device)
                test_img = sample['test'].to(device)
                labels = (sample['label']>0).squeeze(1).type(torch.LongTensor).to(device)
                #labels = (sample['label']>0).type(torch.LongTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss                    
                    outputs = model([reference_img, test_img])
                    #print('got some out')
                    #print(outputs.shape)
                    #print(labels.shape)
                    # Calculate Loss
                    loss = criterion(outputs, labels)

                    # Get the correct class by looking for the max value across channels
                    _, preds = torch.max(outputs, 1)
                    
                    # Calculate metric during evaluation
                    if phase == 'val':
                        dice_value = seg_metrics.iou_segmentation(preds.squeeze(1).type(torch.LongTensor), (labels>0).type(torch.LongTensor))
                        list_dice_val.append(dice_value.item())
                        val_iou_history.append(dice_value.item())
                        epoch_iou = val_iou_history[-1]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * reference_img.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    if iterations % 100 == 0:
                        # Calculate 1/10th of batch size
                        num_imgs = reference_img.shape[0] // 10
                        #writer.add_images('/run/preds', preds[0:num_imgs].unsqueeze(1), iterations)
                        #writer.add_images('/run/labels', labels[0:num_imgs].unsqueeze(1), iterations)
                    iterations += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            #writer.add_scalar('epoch/loss_' + phase, epoch_loss, epoch)
            #if phase == 'val':
                #writer.add_scalar('metrics/iou_val', np.mean(list_dice_val), epoch)
    
            
            # Update Scheduler if training loss doesn't change for patience(2) epochs
            if phase == 'train':
                sc_plt.step(epoch_loss)
                
                # Get current learning rate (To display on Tensorboard)
                for param_group in optimizer.param_groups:
                    curr_learning_rate = param_group['lr']
                    #writer.add_scalar('epoch/learning_rate_' + phase, curr_learning_rate, epoch)

            # deep copy the model and save if accuracy is better
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            '''if phase == 'val':
                val_acc_history.append(epoch_acc) '''       
    
    print('Best val iou: {:4f}'.format(best_iou))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_iou_history