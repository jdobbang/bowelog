#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import os
import cv2
import glob
import numpy as np

def RPN_LOAD():
    img_pth = glob.glob('../raw_dataset/*.jpg')
    # YOLO 가중치 파일과 CFG 파일 로드
    YOLO_net = cv2.dnn.readNet("../weights/bowel.weights","../weights/bowel.cfg")
    # YOLO NETWORK 재구성
    classes = []
    with open("../weights/bowel.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]
    
    return YOLO_net, output_layers


def RPN_detector(img,model,layers):
    
    h, w, c = img.shape
    # YOLO 입력
    blob = cv2.dnn.blobFromImage(img, 0.00392, (640, 640), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    outs = model.forward(layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.05, 0.45)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]            
            # 경계상자와 클래스 정보 이미지에 입력
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            return (x,y,w,h),1

    return (0,0,w,h),0
            
            
            
            
            
            
            