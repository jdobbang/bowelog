import os
from PIL import Image
from classifier_bristol import Classifier
from segmentor import Segmentor
from bbox_from_mask import BoundingBox
import torch

class Inference:

    def __init__(self):

        '''

        Main module that ties all the other relevant models (Segmentor, BoundingBox, Classifier)

        '''

        # GLOBAL CONSTANTS
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration

        # model paths
        CLASSIFIER_DIR = 'saved_models'
        CLASSIFIER_NAME = 'inference_resnet18_layers_6_opt_SGD_lr_0.0001_wd_0.1.pth'
        classifier_path = os.path.join(CLASSIFIER_DIR, CLASSIFIER_NAME)

        SEG_DIR = 'saved_models'
        SEG_PATH = 'AUGGI_SEGNET_STATE_2018_11_28_5_7_NEPOCHS_100_TRAINAVGLOSS_13_8_TESTAVGLOSS_13_7.pth'
        seg_path = os.path.join(SEG_DIR, SEG_PATH)

        # ------------- Instantiate Models -------------  #

        # Instantiate Seg, Boxer and Classifier models

        self.segmentor = Segmentor(seg_path)
        self.boxer = BoundingBox()
        self.classifier = Classifier(classifier_path)

        # ----------------------------------------------  #

    def predict(self, image):

        # ------------ Run full inference on image -----  #

        binary_mask = self.segmentor.segment(image)  # get binary mask (sending rgb PIL)
        x_min, y_min, x_max, y_max = self.boxer.get_box(binary_mask)  # get bbox, sending grayscale PIL image

        # classify the (original) image, pass bboxes, receive a bristol prediction
        bristol_pred = self.classifier.classify(image, x_min, y_min, x_max, y_max)

        return bristol_pred





















