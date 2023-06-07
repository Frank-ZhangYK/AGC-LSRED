# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 19:50:18 2023

@author: DELL
"""
import numpy as np
import torch
import torch.nn as nn


class ae(nn.Module):
    def __init__(self):
        super(ae, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride=1, padding=1)        
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels = 256, out_channels =256, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()


    def forward(self, x):
        #encoder
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        res1 = x2
        x2 = self.max_pooling(x2)  
        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x4 = self.conv4(x3)
        x4 = self.relu(x4)
        res2 = x4
        x4 = self.max_pooling(x4)
        x5 = self.conv5(x4)
        x5 = self.relu(x5)
        x6 = self.conv6(x5)
        x6 = self.relu(x6)
        res3 = x6
        x7 = self.conv7(x6)
        x7 = self.relu(x7)
        x8 = self.conv8(x7)
        x8 = self.relu(x8)
        
        #decoder
        x9 = self.conv9(x8)
        x9 = self.relu(x9)
        x10 = self.conv10(x9)
        x10 = x10 + res3
        x10 = self.relu(x10)
        x11 = self.conv11(x10)
        x11 = self.relu(x11)
        x12 = self.conv12(x11)
        x12 = self.relu(x12)
        x13 = self.deconv1(x12)
        x13 = x13 + res2
        x13 = self.relu(x13)
        x14 = self.conv13(x13)
        x14 = self.relu(x14)
        x15 = self.conv14(x14)
        x15 = self.relu(x15)
        x16 = self.deconv2(x15)
        x16 = x16 + res1
        x16 = self.relu(x16)
        x17 = self.conv15(x16)
        x17 = self.relu(x17)
        x18 = self.conv16(x17)
        x18 = self.relu(x18)
        return x2, x4, x6, x8, x10, x13, x16, x18

