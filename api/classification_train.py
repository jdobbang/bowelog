# -*- coding: utf-8 -*-
from __future__ import print_function, division
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

#%% Train Model
# 모델 학습
def train_model(model, criterion, optimizer, scheduler, dataloaders,num_epochs=10):
    # 파리미터 초기화
    since = time.time()
    # 가장 높은 값을 기준으로 산출하기 위한 DEEP COPY
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[TEST])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        # 학습 모드
        model.train()

        for i, data in enumerate(dataloaders[TRAIN]):
            # if i % 100 == 0:
            print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)

            inputs, labels = data
            if 1: # GPU 사용
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: # CPU 사용
                inputs, labels = Variable(inputs), Variable(labels)
            # 역전파 초기화
            optimizer.zero_grad()
            outputs = model(inputs)
            # 출력값 기반 에러 연산
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            # 업데이트
            loss.backward()
            optimizer.step()
            # 검증을 위한 지표
            loss_train += loss.data
            acc_train += torch.sum(preds == labels.data)
            # 초기화
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_train / dataset_sizes[TRAIN]
        avg_acc = acc_train / dataset_sizes[TRAIN]
        # 테스트 모드
        model.train(False)
        model.eval()

        for i, data in enumerate(dataloaders[TEST]):
            inputs, labels = data
            if 1: # GPU 사용
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else: # CPU 사용
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
            # 초기화 (wirh opti... : 로 변경가능)
            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / dataset_sizes[TEST]
        avg_acc_val = acc_val / dataset_sizes[TEST]

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),'./weights/shape.pt')
    
    return model,best_acc
#%% 학습
#%% 학습 데이터
class_dir = glob.glob('./train_dataset_color/*')
inputs,outputs = [],[]
for i in range(len(class_dir)):
    for j in glob.glob(class_dir[i]+'/*'):
        inputs.append(j)
        outputs.append(i)
print('학습 데이터 수 : {}'.format(len(inputs)))
print('학습 클래수 수 : {}'.format(i+1))

from sklearn.model_selection import KFold
import shutil

#%% K-FOLD 정의
nb_split = 10 # 분할 개수
KF = KFold(n_splits=nb_split, shuffle=True)
acc = []
try:
    shutil.rmtree('./k-fold_Test')
    shutil.rmtree('./k-fold_Train')
except:pass
for train_idx, valid_inx in KF.split(inputs):
    
    for img in train_idx:

        from_ = inputs[img]
        to_ = './k-fold_Train/{}'.format(outputs[img])
        try:
            shutil.copy(from_,to_+'/')
        except:
            os.makedirs('./k-fold_Train/{}'.format(outputs[img]))
            shutil.copy(from_,to_+'/')
            
            
    for img in valid_inx:
        try:
            from_ = inputs[img]
            to_ = './k-fold_Test/{}'.format(outputs[img])
            shutil.copy(from_,to_+'/')
        except:
            os.makedirs('./k-fold_Test/{}'.format(outputs[img]))
            shutil.copy(from_,to_+'/')
    #%% Dataset loader
    data_dir = ''
    TRAIN = 'k-fold_Train'
    TEST = 'k-fold_Test'
    data_transforms = {
    
        TRAIN: transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
        TEST: transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in [TRAIN, TEST]
    }
    
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=8,
            shuffle=True
        )
        for x in [TRAIN,TEST]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, TEST]}
    #%% Load Model
    # 사전 학습된 모델 불러오기
    model = load_vgg()
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

    best_model,score = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,dataloaders,num_epochs=15)
    acc.append(score)
    # 테스트 | 학습 데이터 리프레시
    shutil.rmtree('./k-fold_Test')
    shutil.rmtree('./k-fold_Train')
    break


