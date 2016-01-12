from flask import Flask, request, redirect, url_for, render_template
import cPickle as pickle
import json
import os
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
        f_name = str(uuid.uuid4()) + '.wav'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
        tmp_file_path = app.config['UPLOAD_FOLDER'] + f_name
        X = single_file_featurization(tmp_file_path)
        y_pred = svm.predict(X)
        print y_pred
        return render_template('result.html', pred = y_pred)
    else:
        abort(make_response("File extension not acceptable", 400))

@app.route('/result', methods= ['GET'])
def result():
    return render_template('result.html')


if __name__ == '__main__':
    with open('static/model/svm.pkl', 'rb') as f1:
        svm = cPickle.load(f1)
    with open('static/model/lda.pkl', 'rb') as f2:
        lda = cPickle.load(f2)
    with open('static/model/ss.pkl', 'rb') as f3:
        ss = cPickle.load(f3)
    app.run(host='0.0.0.0', port=7777, debug=True)