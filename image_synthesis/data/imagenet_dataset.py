from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from image_synthesis.utils.misc import instantiate_from_config

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class ImageNetDataset(Dataset):
    def __init__(self, data_root, input_file, phase = 'train', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.root = os.path.join(data_root, phase)
        input_file = os.path.join(data_root, input_file)
        
        temp_label = json.load(open('image_synthesis/data/imagenet_class_index.json', 'r'))
        self.labels = {}
        for i in range(1000):
            self.labels[temp_label[str(i)][0]] = i
        self.A_paths = []
        self.A_labels = []
        with open(input_file, 'r') as f:
            temp_path = f.readlines()
        for path in temp_path:
            label = self.labels[path.split('/')[0]]
            self.A_paths.append(os.path.join(self.root, path.strip()))
            self.A_labels.append(label)

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        # if self.transform is not None:
        A = self.transform(A)['image']
        A_label = self.A_labels[index % self.A_size]
        data = {
                'image': np.transpose(A.astype(np.float32), (2, 0, 1)),
                'label': A_label,
                }
        return data
