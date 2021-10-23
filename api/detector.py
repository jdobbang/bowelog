#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: deep.i inc.

object detection [YOLO V4 608 MODEL AP 99%]
"""
import sys
sys.path.append('../')
import cv2
import numpy as np
import glob

img_pth = glob.glob('./raw_dataset/*.jpg')
# YOLO 가중치 파일과 CFG 파일 로드
YOLO_net = cv2.dnn.readNet("./weights/bowel.weights","weights/bowel.cfg")

# YOLO NETWORK 재구성
classes = []
with open("weights/bowel.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]

for j in img_pth:
    # READ IMAGE
    frame = cv2.imread(j)
    h, w, c = frame.shape

    # YOLO 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 608), (0, 0, 0),
    True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

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
            label = str(classes[class_ids[i]])
            score = confidences[i]

            # 경계상자와 클래스 정보 이미지에 입력
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, 
            (255, 255, 255), 1)

    cv2.imshow("DEEP.EYE", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()