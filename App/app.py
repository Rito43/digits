from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Helper function to compare two images
def compare_images(image1, image2):
    # Load and preprocess the images
    img1 = cv2.imread(image1, 0)
    img2 = cv2.imread(image2, 0)

    # Check if images are the same shape
    if img1.shape != img2.shape:
        return False

    # Compute the absolute difference between the images
    diff = cv2.absdiff(img1, img2)

    # Define a threshold for similarity (adjust as needed)
    threshold = 30

    # Compare the images based on the threshold
    return (np.sum(diff) / 255) / (img1.shape[0] * img1.shape[1]) <= threshold

@app.route('/compare_images', methods=['POST'])
def compare_images_route():
    data = request.json

    if 'image1' not in data or 'image2' not in data:
        return jsonify({'error': 'Missing image1 or image2 in the request.'}), 400

    image1 = data['image1']
    image2 = data['image2']

    result = compare_images(image1, image2)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
