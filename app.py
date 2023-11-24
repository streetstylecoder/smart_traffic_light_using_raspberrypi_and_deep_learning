from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np

app = Flask(__name__)

STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

app.config['STATIC_FOLDER'] = STATIC_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_cars(file_path):
    # Car detection logic
    image = cv2.imread(file_path)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    dilated = cv2.dilate(blur, np.ones((3, 3)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    car_cascade_src = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    cars = car_cascade.detectMultiScale(closing, 1.1, 1)

    count = 0
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

    # Save the result with rectangles to a new file in the 'static' folder
    result_filename = 'result_' + os.path.basename(file_path)
    result_path = os.path.join(app.config['STATIC_FOLDER'], result_filename)
    cv2.imwrite(result_path, image)

    return count, result_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file in the 'static' folder
        original_filename = file.filename
        filepath = os.path.join(app.config['STATIC_FOLDER'], original_filename)
        file.save(filepath)

        # Process the image and get the count
        count, result_filename = detect_cars(filepath)

        return render_template('index.html', count=count, original_filename=original_filename, result_filename=result_filename)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
