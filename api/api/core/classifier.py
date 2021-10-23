#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import glob
import cv2
import numpy as np
import time
import psutil

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

def CNN_LOAD(mode):

    # load model
    model = load_vgg()
    model.load_state_dict(torch.load('../weights/weight_cpu/{}.pt'.format(mode), map_location=torch.device('cpu')))
    # cuda mode
    model.eval()
    return model

def CNN_classifier(image,model):
    
    my_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    # For reversing the operation:
    im_np = np.asarray(im_pil)
    
    img_ = my_transforms(im_pil).unsqueeze(0)
    _,idx = torch.max(model(img_),1)
    return idx