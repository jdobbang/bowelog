import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image

class Classifier:

    def __init__(self, PATH):

        N_LABELS = 7  # Bristol numbers

        self.model = torchvision.models.resnet18(pretrained=False)  # load blank model

        # change number of output layers in last layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, N_LABELS)

        checkpoint = torch.load(PATH)  # load checkpoint
        self.model.load_state_dict(checkpoint)   # load state dict (weights)
        self.model.eval()  # set to eval mode

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def classify(self, image, x_min, y_min, x_max, y_max):
        '''
        Needs to receive original full PIL image and 4 bounding box coordinates in a tuple.
        It will crop it and classify, returning Bristol prediction (int)

        '''

        SIZE = 224

        # need to resize to 224 x 224 first (since bbox are relative to this size)
        image = image.resize((SIZE, SIZE), Image.ANTIALIAS)

        # will crop the bbox out
        image = image.crop((x_min, y_min, x_max, y_max))

        # reshape, convert to tensor and normalize by ImageNet values
        resize = transforms.Resize((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # define transforms
        image = resize(image)
        image = to_tensor(image)
        image = normalize(image)

        image = torch.unsqueeze(image, 0)  # add 1 for batch size
        
        # put image on appropriate device
        image = image.to(self.device)
        
        output = self.model(image)  # forward prop

        _, pred = torch.max(output.data, 1)  # get top pred

        # also add 1 for array index offset
        return  pred.item() + 1
