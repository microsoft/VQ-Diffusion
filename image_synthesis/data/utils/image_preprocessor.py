import albumentations
import random
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from torchvision import transforms as trans

class DalleTransformerPreprocessor(object):
    def __init__(self, 
                 size=256,
                 phase='train',
                 additional_targets=None):
    
        self.size = size 
        self.phase = phase
        # ddc: following dalle to use randomcrop
        self.train_preprocessor = albumentations.Compose([albumentations.RandomCrop(height=size, width=size)],
                                                   additional_targets=additional_targets)
        self.val_preprocessor = albumentations.Compose([albumentations.CenterCrop(height=size, width=size)],
                                                   additional_targets=additional_targets)                                                   


    def __call__(self, image, **kargs):
        """
        image: PIL.Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)) 

        w, h = image.size
        s_min = min(h, w)

        if self.phase == 'train':
            off_h = int(random.uniform(3*(h-s_min)//8, max(3*(h-s_min)//8+1, 5*(h-s_min)//8)))
            off_w = int(random.uniform(3*(w-s_min)//8, max(3*(w-s_min)//8+1, 5*(w-s_min)//8)))
            # import pdb; pdb.set_trace()
            image = image.crop((off_w, off_h, off_w + s_min, off_h + s_min))

            # resize image
            t_max = min(s_min, round(9/8*self.size))
            t_max = max(t_max, self.size)
            t = int(random.uniform(self.size, t_max+1))
            image = image.resize((t, t))
            image = np.array(image).astype(np.uint8)
            image = self.train_preprocessor(image=image)   #randomcrop (size,size)
        else:
            if w < h:
                w_ = self.size 
                h_ = int(h * w_/w)
            else:
                h_ = self.size
                w_ = int(w * h_/h)
            image = image.resize((w_, h_))
            image = np.array(image).astype(np.uint8)
            image = self.val_preprocessor(image=image)
        return image


class ImageNetTransformerPreprocessor(object):
    def __init__(self, 
                 size=256,
                 phase='train',
                 additional_targets=None):
    
        self.size = size 
        self.phase = phase
        # ddc: following dalle to use randomcrop
        self.train_preprocessor = albumentations.Compose([albumentations.RandomCrop(height=size, width=size)],
                                                   additional_targets=additional_targets)
        self.val_preprocessor = albumentations.Compose([albumentations.CenterCrop(height=size, width=size)],
                                                   additional_targets=additional_targets)                                                   


    def __call__(self, image, **kargs):
        """
        image: PIL.Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)) 

        w, h = image.size
        s_min = min(h, w)

        if self.phase == 'train':
            if w < h:
                w_ = self.size
                h_ = int(h * w_/w)
            else:
                h_ = self.size
                w_ = int(w * h_/h)
            image = image.resize((w_, h_))
            image = np.array(image).astype(np.uint8)
            image = self.train_preprocessor(image=image)
        else:
            if w < h:
                w_ = self.size 
                h_ = int(h * w_/w)
            else:
                h_ = self.size
                w_ = int(w * h_/h)
            image = image.resize((w_, h_))
            image = np.array(image).astype(np.uint8)
            image = self.val_preprocessor(image=image)
        return image

