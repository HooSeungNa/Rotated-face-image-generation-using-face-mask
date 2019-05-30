import numpy as np
import os 
import cv2
import torch
from torch.utils.data import Dataset
import scipy.io
import dlib
import torchvision.transforms as transforms
from skimage import io, transform

import matplotlib.pyplot as plt

#using face mask
class ImageDataset(Dataset):
    def __init__(self, fnames, transform=None):
        
        #label 값을 불러온다
        self.ia=np.load(fnames[0])
        self.ib=np.load(fnames[1])
        self.pa=np.load(fnames[2])
        self.pb=np.load(fnames[3])
        self.pam=np.load(fnames[4])
        self.pbm=np.load(fnames[5])
        self.gray_b=np.load(fnames[6])
        self.transform=transform
    def __len__(self):
        return len(self.ia)
#         return 1000
    
    def __getitem__(self, i):
        ia=self.ia[i]
        ib=self.ib[i]
        gray_b = self.gray_b[i]
        pa=self.pa[i]
        pb=self.pb[i]
        pam=self.pam[i]
        pbm=self.pbm[i]
        
        sample = [ia,ib,pa,pb,pam,pbm,gray_b]
        if self.transform:
            sample = self.transform(sample)
        return sample
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        ia,ib,pa,pb,pam,pbm,gray_b = sample[0], sample[1],sample[2],sample[3],sample[4],sample[5],sample[6]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        ia = ia/255
        ia = ia.transpose((2, 0, 1))
        ia = torch.from_numpy(ia)
        
        ib = ib/255
        ib = ib.transpose((2, 0, 1))
        ib = torch.from_numpy(ib)
        
        pa = pa/255
        pa = pa.transpose((2, 0, 1))
        pa = torch.from_numpy(pa)
        
        pb = pb/255
        pb = pb.transpose((2, 0, 1))
        pb = torch.from_numpy(pb)
        
        pam = pam/255
        pam=np.expand_dims(pam,-1)
        pam = pam.transpose((2, 0, 1))
        pam = torch.from_numpy(pam)
        
        pbm = pbm/255
        pbm=np.expand_dims(pbm,-1)
        pbm = pbm.transpose((2, 0, 1))
        pbm = torch.from_numpy(pbm)
        
        
        gray_b = gray_b/255
        gray_b=np.expand_dims(gray_b,-1)
        gray_b = gray_b.transpose((2, 0, 1))
        gray_b = torch.from_numpy(gray_b)
        
        result=[ia.type(torch.FloatTensor),ib.type(torch.FloatTensor),pa.type(torch.FloatTensor),
                pb.type(torch.FloatTensor),pam.type(torch.FloatTensor),pbm.type(torch.FloatTensor),gray_b.type(torch.FloatTensor)]
        return result