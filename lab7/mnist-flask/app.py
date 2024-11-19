#This code is based on a gitbub project available at https://github.com/akashdeepjassal/mnist-flask
from flask import Flask, render_template, request
from imageio import imread, imsave
from skimage.transform import resize
import numpy as np
import keras.models
import re
import base64

import sys 
import os
# append the model path
sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit
    x = imread('output.png')
	# covert image to grey scale & convert it to numby array
    x = x[:,:,0]
    x = np.invert(x)
	# resize the image to the training size
    x = resize(x,(28,28))
    print(x.shape)

    #reshape image data for use in neural network
    x = x.reshape((1,28,28,1))
    # load the model 
    from tensorflow.keras.models import load_model
    model = load_model('model\model.h5')    
    print("Loaded Model from disk")
	# predict the result
    out = model.predict(x)
    print(out)
	# get the max value & change it to string
    print(np.argmax(out, axis=1))
    response = np.array_str(np.argmax(out, axis=1))
    return response 
	
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
	app.run(debug=True)