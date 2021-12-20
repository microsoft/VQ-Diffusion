from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from image_synthesis.utils.misc import instantiate_from_config
import torchvision.datasets as datasets

class FFHQDataset(datasets.ImageFolder):
    def __init__(self, data_root, im_preprocessor_config):
        self.img_preprocessor = instantiate_from_config(im_preprocessor_config)
        super(FFHQDataset, self).__init__(root=data_root) 
 
    def __getitem__(self, index):
        # image_name = self.imgs[index][0].split('/')[-1]
        image = super(FFHQDataset, self).__getitem__(index)[0]
        image = self.img_preprocessor(image=np.array(image).astype(np.uint8))['image']
        data = {
                'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                }
        return data

    
