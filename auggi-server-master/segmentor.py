from models.segnet import SegNet
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


class Segmentor:

    '''

    Instantiates a SegNet model, loads the trained auggi model, and can process
    a PIL image, returning a binary mask.

    '''


    def __init__(self, seg_path):

        '''

        Loads the SegNet model with auggi trained weights

        '''

        N_LABELS = 1 # Binary Classifier

        # Segmentation model needs to load here and set to eval
        self.model = SegNet(input_nbr=3, label_nbr=N_LABELS)
        self.model.load_from_filename(seg_path) # load auggi trained weights
        self.model.eval() # set to eval mode

        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # move to right device

    def segment(self, image):
        '''
        Needs to receive a RGB PIL image, returns a PIL image binary mask

        '''

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
        
        mask = self.model(image)  # forward prop

        mask_as_img = F.to_pil_image(mask[0])  # don't forget to grab the first entry of mask (4d tensor)

        # return the binary mask PIL image
        return mask_as_img
