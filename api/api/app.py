import flask
from flask import Flask, render_template 
from flask import request
app=Flask(__name__)

#
import os
import copy
import numpy as np
from datetime import datetime

# Image Processing
import cv2
from PIL import Image

# Deep learning
from core.rpn import RPN_LOAD, RPN_detector
from core.classifier import CNN_LOAD, CNN_classifier
from core.imgprocessing import Blood
# from common import get_tensor
# from grade import get_grade
import io

# RPN LOAD
rpn, rpn_layers = RPN_LOAD()
s_model = CNN_LOAD('shape')
c_model = CNN_LOAD('color')

shapes = ['hard_lump','hard_div','clay','moist','dry','watery','soft_div']
colors = ['#cf4e11','#000000','#d1c5b4','#dfb337','#985a0e','#2a5f11']

@app.route('/main/', methods=['GET','POST'])
def main():
    
    if request.method == 'GET':
        return render_template('main.html', value='hello',image_file = '바울로그라피_영어.png')
    
    if request.method == 'POST':
        # file load
        file = request.files["file"]
        image = file.read()
        # decoding
        encoded_img = np.fromstring(image, dtype = np.uint8)
        image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        dimg = copy.deepcopy(image)
        # object detection
        rpn_bbox,ret = RPN_detector(image,rpn,rpn_layers)
        if ret == 0:
            r_image,s_result,c_result,b_area,b_score,s_area = 0,0,0,0,0,0
        else :    
            x,y,w,h = rpn_bbox
            r_image = image[abs(y):abs(y)+h,abs(x):abs(x)+w]
            # classification (shape)
            s_result = shapes[CNN_classifier(r_image,s_model)]
            # classification (color)
            c_result = colors[CNN_classifier(r_image,c_model)]
            # blood area check
            b_score, s_area, b_area = Blood(r_image)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
            
        cv2.imwrite('./static/image/upload.jpg',image)
        #r_image,s_result,c_result,b_area,b_score,s_area = 0,0,0,0,0,0
        
        # DB 
        client_ip = flask.request.remote_addr
        client_date = datetime.today().strftime("%Y%m%d%H%M%S")

        client_info = "./DB/[{}]-{}-{}-{}-{}.jpg".format(client_ip,client_date,ret,s_result,c_result)  
        cv2.imwrite(client_info,dimg )

        return render_template('result.html', objects = ret, bbox = rpn_bbox, shape = s_result, color = c_result,
                               blood_score = b_score, size = s_area, blood = b_area,  img = image, image_file = '/image/upload.jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='8080',debug=False)
    