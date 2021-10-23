#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:35:10 2021

@author: deep
"""


from __future__ import print_function, division
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import glob
import cv2
import numpy as np
import time
import psutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import glob
#%% Load Model
# 사전 학습된 모델 불러오기
def load_vgg():
    model = models.vgg16(pretrained=True)
    # CNN Pre-trained 가중치를 그대로 사용할때
    for param in model.features.parameters():
        param.require_grad = False
    features = model.features    
    num_features = model.classifier[6].in_features
    # # FC 레이어 추가
    features = list(model.classifier.children())[:-1]   # 출력 레이어 제거
    features.extend([nn.Linear(num_features,7)])        # 출력 레이어 변경
    model.classifier = nn.Sequential(*features)         # 분류기 업데이트
    return model
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
    return [np.array(inputs),np.array(outputs),labels]

mode = 'shape'
# load data
inputs,outputs,classes = setDataset(mode)
# load model
model = load_vgg()
d = torch.device('cuda')
model.load_state_dict(torch.load('./weights/{}.pt'.format(mode),d))
# cuda mode
model.eval()

for ii,i in enumerate(inputs):
    
    img = Image.open(i)
    my_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
    
    img_ = my_transforms(img).unsqueeze(0)
    _,idx = torch.max(model(img_),1)
    img.show()
    print('predict {} : {} target {} : {}'.format(mode,
                                                  classes[idx.item()].split('/')[-1],mode,classes[outputs[ii]].split('/')[-1]))
    
    time.sleep(1)
    
    for proc in psutil.process_iter():
        if proc.name() == "display":
            proc.kill()
    
    
    

