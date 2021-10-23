#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:35:10 2021

@author: deep
"""



import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import glob
import cv2
import numpy as np
import time
import psutil

#%% TEST 데이터
def setDataset(mode):
    class_dir = glob.glob('./train_dataset_{}/*'.format(mode))
    inputs,outputs = [],[]
    labels = []
    for i in range(len(class_dir)):
        labels.append(class_dir[i].split('\\')[-1])
        for j in glob.glob(class_dir[i]+'/*.jpg'):
            inputs.append(j)
            outputs.append(i)
    print('학습 데이터 수 : {}'.format(len(inputs)))
    print('학습 클래수 수 : {}'.format(i+1))
    return [np.array(inputs),np.array(outputstputs),labels]

mode = 'color'
# load data
inputs,outputs,classes = setDataset(mode)
# load model
model = torch.load('../weights/{}.pt'.format(mode),'gpu')
# cuda mode
model.eval()

for ii,i in enumerate(inputs):
    
    img = Image.open(i)
    my_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
    
    img_ = my_transforms(img).unsqueeze(0)
    _,idx = torch.max(model(img_.cuda()),1)
    img.show()
    print('predict {} : {} target {} : {}'.format(mode,
                                                  classes[idx.item()].split('/')[-1],mode,classes[outputs[ii]].split('/')[-1]))
    
    time.sleep(1)
    
    for proc in psutil.process_iter():
        if proc.name() == "display":
            proc.kill()
    
    
    

