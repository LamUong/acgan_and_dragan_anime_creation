from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
# Ignore warnings
import warnings
import time
warnings.filterwarnings("ignore")
class anime_face(Dataset):
    def __init__(self, attr_pickle, image_dir, transform=None):
        """
        Args:
            attr_pickle (string): Path to the pickle file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attr_pickle = pickle.load( open( attr_pickle, "rb" ) )
        
        self.image_dir = image_dir
        
        self.image_names = os.listdir(image_dir)
        
        self.image_index_to_name = {}
        
        for i in range(0,len(self.image_names)):
            self.image_index_to_name[i] = self.image_names[i]

        self.transform = transform

    def __len__(self):
        return len(self.attr_pickle)

    def __getitem__(self, idx):
        img_name = self.image_dir + self.image_names[idx]
        
        image = io.imread(img_name)

        tags = np.array(self.attr_pickle[self.image_names[idx]]).astype('float')
        tags = torch.FloatTensor(tags)

        if self.transform:
            image = self.transform(image)

        return image, tags



