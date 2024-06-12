# Pytorch stuff
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import torch.utils.data as utils
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

SMOOTH = 1e-6
# Expect outputs and labels to have same shape (ie: torch.Size([batch:1, 224, 224])), and type long
def iou_segmentation(outputs: torch.Tensor, labels: torch.Tensor):    
    # Will be zero if Truth=0 or Prediction=0 
    intersection = (outputs & labels).float().sum((1, 2))    
    # Will be zzero if both are 0   
    union = (outputs | labels).float().sum((1, 2))          
    
    # We smooth our devision to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)      
    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded