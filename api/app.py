import matplotlib.pyplot as plt
import sys
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pdb
from joblib import dump,load
import numpy as np
# import skimage
# from skimage.transform import resize
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

current_directory = os.getcwd()

folder_path = os.path.join(current_directory, 'models')
extension = '.joblib'
all_files = os.listdir(folder_path)
matching_files = [file for file in all_files if file.endswith(extension)]

file_path = os.path.join(folder_path, matching_files[0])

model = load(file_path)
 
@app.route('/predict', methods=['POST'])
def compare_digits():
    try:
        data = request.get_json()  
        image1 = data.get('image1', [])
        image2 = data.get('image2', [])

        digit1 = predict_digit(image1)
        digit2 = predict_digit(image2)

        result = digit1 == digit2

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})
    
def predict_digit(image):
    try:
        img_array = np.array(image, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        return digit
    except Exception as e:
        return str(e)


if __name__ == '_main_':
    app.run()