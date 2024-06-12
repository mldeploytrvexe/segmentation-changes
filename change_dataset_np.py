from torch.utils.data import Dataset
import numpy as np
import pickle
import torchvision
import random
import helper_augmentations
from PIL import Image

class ChangeDatasetNumpy(Dataset):
    """ChangeDataset Numpy Pickle Dataset"""

    def __init__(self, pickle_file, transform=None):
    
        
        f = open(pickle_file,"rb")
        self.data_dict= pickle.load(f)
        f.close()          
    
        self.transform = transform

    def __len__(self):
        return len(self.data_dict['label'])

    def __getitem__(self, idx):        
     
        reference_PIL = self.data_dict['reference'][idx]
        reference_PIL_img = Image.open(f'{reference_PIL}').convert('RGB')
        test_PIL = self.data_dict['test'][idx]
        test_PIL_img = Image.open(test_PIL).convert('RGB')
        label_PIL = self.data_dict['label'][idx]
        label_PIL_img = Image.open(label_PIL)
        sample = {'reference': reference_PIL_img, 'test': test_PIL_img, 'label': label_PIL_img}        

        
        if self.transform:
            trf_reference = sample['reference']
            trf_test = sample['test']          
            trf_label = sample['label']
            
            for t in self.transform.transforms:
                if (isinstance(t, helper_augmentations.SwapReferenceTest)) or (isinstance(t, helper_augmentations.JitterGamma)):
                    trf_reference, trf_test = t(sample)
                else:
                   
                    trf_reference = t(trf_reference)
                    trf_test = t(trf_test)
                

                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        trf_label = t(trf_label) * 255.0
                    else:
                        if not isinstance(t, helper_augmentations.SwapReferenceTest):
                            if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                                if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                    if not isinstance(t, helper_augmentations.JitterGamma):
                                        trf_label = t(trf_label)
                              
            sample = {'reference': trf_reference, 'test': trf_test, 'label': trf_label}

        return sample