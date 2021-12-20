from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from image_synthesis.utils.misc import instantiate_from_config
from tqdm import tqdm
import pickle

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class Cub200Dataset(Dataset):
    def __init__(self, data_root, phase = 'train', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.image_folder = os.path.join(data_root, 'images')
        self.root = os.path.join(data_root, phase)
        pickle_path = os.path.join(self.root, "filenames.pickle")
        self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")

        self.num = len(self.name_list)

        # load all caption file to dict in memory
        self.caption_dict = {}
        
        for index in tqdm(range(self.num)):
            name = self.name_list[index]
            this_text_path = os.path.join(data_root, 'text', 'text', name+'.txt')
            with open(this_text_path, 'r') as f:
                caption = f.readlines()
            self.caption_dict[name] = caption

        print("load caption file done")


    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        name = self.name_list[index]
        image_path = os.path.join(self.image_folder, name+'.jpg')
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        caption_list = self.caption_dict[name]
        caption = random.choice(caption_list).replace('\n', '').lower()
        
        data = {
                'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                'text': caption,
        }
        
        return data
