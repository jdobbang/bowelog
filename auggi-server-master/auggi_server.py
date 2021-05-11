#from flask import Flask
import os
from full_inference import Inference
from PIL import Image



#app = Flask(__name__)
 
#@app.route("/")
def predict():

    # image path - for testing, change this
    image_dir = 'test_images'
    image_name = '8bb3bb82-ed62-4152-aa9e-b15934a17f4e.jpg'


    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path)  # open image with PIL

    infer = Inference()  # create inference object
    bristol = infer.predict(image)  # call predict(), passing a PIL image
    
    print(str(bristol))
    return str(bristol)

#if __name__ == "__main__":
    #app.run()