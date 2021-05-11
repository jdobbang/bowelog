import numpy as np
from test_convolution import convolution



def smoothing(image,verbose):
    kernel = np.array([[2,4,5,4,2],
                      [4,9,12,9,4],
                      [5,12,15,12,5],
                      [4,9,12,9,4],
                      [2,4,5,4,2]],dtype=np.float64)/159
   
    return convolution(image, kernel, average=True, verbose=verbose)