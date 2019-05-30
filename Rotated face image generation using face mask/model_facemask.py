import torch 
import torch.nn as nn
import cv2
import numpy as np 
import time
import math
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from unet_parts import *

########Light cnn 29 layers##########
class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])
class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x
class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out    
class network_29layers(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers, self).__init__()
        self.conv1  = mfm(1, 48, 5, 1, 2)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc     = mfm(8*8*128, 256, type=0)
        self.fc2    = nn.Linear(256, num_classes)
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)
        pool_x=x
        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)
        return pool_x, fc

class UNet_generator(nn.Module):
    def __init__(self):
        super(UNet_generator, self).__init__()
        self.inc = inconv(13, 128)
        self.down1 = down(128, 256)
        self.down2 = down(256, 512)
        self.down3 = down(512, 512)
#         self.down4 = down(1024, 1024)
#         self.down5 = down(2048,2048)
#         self.up1 = up(4096, 1024)
#         self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 256)
        self.outc = outconv(256, 3)
        self.outc_1ch=outconv_one_channel(256, 1)
    def forward(self, in1,in2,in3):
        x=torch.cat((in1,in2,in3),1)
#     def forward(self, in1):
#         x=in1
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x6 = self.down5(x5)
#         x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
#         x = self.up5(x, x1)
        x_3ch = self.outc(x)
        x_1ch = self.outc_1ch(x)
        
        return F.sigmoid(x_3ch),F.sigmoid(x_1ch)
#         return x_3ch,x_1ch,e_out
#         return F.relu(x_3ch),F.relu(x_1ch),e_out
#         return F.tanh(x_3ch),F.tanh(x_1ch),e_out
    
class Discriminator1(nn.Module):
    def __init__(self, input_dim=6, dim=64):
        super(Discriminator1, self).__init__()

        self.conv1  = nn.Sequential(nn.Conv2d(input_dim, dim, 3, 2, 0),
                                    nn.BatchNorm2d(dim),
                                    nn.ReLU())
        self.conv2  = nn.Sequential(nn.Conv2d(dim, dim*2, 3, 2, 0),
                                    nn.BatchNorm2d(dim*2),
                                    nn.ReLU())
        self.conv3  = nn.Sequential(nn.Conv2d(dim*2, dim*4, 3, 2, 0),
                                    nn.BatchNorm2d(dim*4),
                                    nn.ReLU())
        self.conv4  = nn.Sequential(nn.Conv2d(dim*4, dim*8,3, 2, 0),
                                    nn.BatchNorm2d(dim*8),
                                    nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(25088, 1),nn.Sigmoid())
    
    def forward(self, x1,x2):
        x = torch.cat((x1,x2),1)
#     def forward(self, x1):
#         x = x1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        return x.view(1,-1)
class Discriminator2(nn.Module):
    def __init__(self, input_dim=8, dim=64):
        super(Discriminator2, self).__init__()
        self.conv1  = nn.Sequential(nn.Conv2d(input_dim, dim, 3, 2, 0),
                                    nn.BatchNorm2d(dim),
                                    nn.ReLU())
        self.conv2  = nn.Sequential(nn.Conv2d(dim, dim*2, 3, 2, 0),
                                    nn.BatchNorm2d(dim*2),
                                    nn.ReLU())
        self.conv3  = nn.Sequential(nn.Conv2d(dim*2, dim*4, 3, 2, 0),
                                    nn.BatchNorm2d(dim*4),
                                    nn.ReLU())
        self.conv4  = nn.Sequential(nn.Conv2d(dim*4, dim*8,3, 2, 0),
                                    nn.BatchNorm2d(dim*8),
                                    nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(25088, 1),nn.Sigmoid())
    
    def forward(self, x1,x2):
        x = torch.cat((x1,x2),1)
#     def forward(self, x1):
#         x = x1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        return x.view(1,-1)
#loss function
def multi_scale_pixel_wise_loss(target_image,generate_image):
#     target_img=[]
#     generate_img=[]
#     generate_img1=generate_image.reshape(-1, 3, 32, 32)
#     generate_img2=generate_image.reshape(-1, 3, 64, 64)
    
#     target_img1 = target_image.reshape(-1, 3, 32, 32)
#     target_img2 = target_image.reshape(-1, 3, 64, 64)
    
    
#     target_img.append(target_img1)
#     target_img.append(target_img2)
#     target_img.append(target_image)
    
#     generate_img.append(generate_img1)
#     generate_img.append(generate_img2)
#     generate_img.append(generate_image)
    
#     total_loss=0
    l1_loss = nn.L1Loss()
#     for i in range(3):
#         total_loss+=l1_loss(generate_img[i],target_img[i])
#     total_loss=total_loss/3
    total_loss=l1_loss(generate_image,target_image)
    return total_loss

def pose_mask_loss(G1,x_target,mask_target):
    l1_loss = nn.L1Loss()
    return l1_loss(G1 * mask_target, x_target * mask_target)

def identity_preserving_loss(D_pool_i_gb,D_pool_i_b,D_fc_i_gb,D_fc_i_b):
    loss_ip=0
    result1=0
    loss=nn.L1Loss()
    
    result1=loss(D_pool_i_gb,D_pool_i_b)
    result2=loss(D_fc_i_gb,D_fc_i_b)
    loss_ip=result1+result2
    return loss_ip
def total_variant_regularization(generated_img):

    loss_tv = torch.mean(torch.abs(generated_img[:,:,:-1,:]-generated_img[:,:,1:,:]))+torch.mean(torch.abs(generated_img[:,:,:,:-1]-generated_img[:,:,:,1:]))
    return loss_tv
def total_loss(loss_pix,loss_ii_adv,loss_pe_adv,loss_ip,loss_tv):
    lam1=10
    lam2=0.1
    lam3=0.1
    lam4=0.02
    lam5=1e-4
    total=loss_pix*lam1+loss_ii_adv*lam2+loss_pe_adv*lam3+loss_ip*lam4+loss_tv*lam5
    return total

    