#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

def Blood(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_threshold, image_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #blood hsv range1
    lower_red = (0,100,10)
    upper_red = (7,255,255)    
    red_1 = cv2.inRange(hsv, lower_red, upper_red)
    
    #blood hsv range2
    lower_red2 = (173,100,10)
    upper_red2 = (180,255,255)
    red_2 = cv2.inRange(hsv,lower_red2,upper_red2)
    
    mask = red_1 + red_2
    img_result = cv2.bitwise_and(img, img, mask = mask)
    
    blood = cv2.cvtColor(img_result, cv2.COLOR_HSV2BGR)
    bloodCount = cv2.cvtColor(blood, cv2.COLOR_BGR2GRAY)
       
    #morphology close + open
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    blood1 = cv2.morphologyEx(bloodCount, cv2.MORPH_CLOSE, kernel2)
    blood2 = cv2.morphologyEx(blood1, cv2.MORPH_OPEN, kernel2)
    ret,blood2th = cv2.threshold(blood2, 20, 255, cv2.THRESH_BINARY)
    blood2c = cv2.Canny(blood2,50,150)
    contours, hierarchy = cv2.findContours(blood2c, 2,1)
    for i in range(len(contours)):
       cv2.drawContours(img, [contours[i]], 0, (0, 0, 255), 2)

    bloodArea = cv2.countNonZero(blood2)
    stoolArea = cv2.countNonZero(255-image_result)
    if round(bloodArea/stoolArea,3) > 1 :
        score = 1.0
    else:
        score = round(bloodArea/stoolArea,3)
    return score, stoolArea,bloodArea

