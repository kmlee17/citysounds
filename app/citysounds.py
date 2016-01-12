from flask import Flask, request, redirect, url_for, render_template
import cPickle as pickle
import json
import os
import subprocess
import uuid
import requests
import socket
import cPickle
from werkzeug import secure_filename
from predict_sound import single_file_featurization

UPLOAD_FOLDER = 'static/tmp/'
ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/import', methods= ['POST']) 
def import_objects():        
    file = request.files['file']
    if file and allowed_file(file.filename):
        #extract content 
        print file
        f_name_orig = str(uuid.uuid4()) + '.wav'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name_orig))
        tmp_file_path = app.config['UPLOAD_FOLDER'] + f_name_orig
        f_name_convert = str(uuid.uuid4()) + '.wav'
        converted_file_path = app.config['UPLOAD_FOLDER'] + f_name_convert
        command = "sox " + tmp_file_path + " -b 16 " + converted_file_path
        print command
        subprocess.call(command, shell=True)
        X = single_file_featurization(tmp_file_path)
        y_pred = svm.predict(X)
        pred_class, img = get_class_and_image(y_pred)
        return render_template('result.html', pred = pred_class, class_image = img)
    else:
        abort(make_response("File extension not acceptable", 400))

def get_class_and_image(y_pred):
    if y_pred == 'dog_bark':
        img = 'static/img/dog.jpg'
        return 'Dog Bark', img
    elif y_pred == 'car_horn':
        img = 'static/img/car.jpg'
        return 'Car Horn', img


if __name__ == '__main__':
    with open('static/model/svm.pkl', 'rb') as f1:
        svm = cPickle.load(f1)
    with open('static/model/lda.pkl', 'rb') as f2:
        lda = cPickle.load(f2)
    with open('static/model/ss.pkl', 'rb') as f3:
        ss = cPickle.load(f3)
    app.run(host='0.0.0.0', port=7777, debug=True)