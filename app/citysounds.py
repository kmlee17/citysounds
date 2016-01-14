from flask import Flask, request, redirect, url_for, render_template
import os
import subprocess
import uuid
import cPickle
from werkzeug import secure_filename
from predict_sound import single_file_featurization

ROOT_FOLDER = '/Users/kevinlee/citysounds/app/'
UPLOAD_FOLDER = ROOT_FOLDER + 'static/tmp/'
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
        f_name_orig = str(uuid.uuid4()) + '.wav'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name_orig))
        tmp_file_path = app.config['UPLOAD_FOLDER'] + f_name_orig
        f_name_convert = str(uuid.uuid4()) + '.wav'
        converted_file_path = app.config['UPLOAD_FOLDER'] + f_name_convert
        command = "sox " + tmp_file_path + " -b 16 " + converted_file_path
        print command
        subprocess.call(command, shell=True)
        X = single_file_featurization(converted_file_path)
        with open(ROOT_FOLDER + 'static/model/svm.pkl', 'rb') as f1:
            svm = cPickle.load(f1)
        y_pred = svm.predict(X)
        pred_class, img = get_class_and_image(y_pred)
        return render_template('result.html', pred = pred_class, class_image = img)
    else:
        abort(make_response("File extension not acceptable", 400))

def get_class_and_image(y_pred):
    if y_pred == 'air_conditioner':
        img = 'static/img/ac.jpg'
        return 'Air Conditioner', img
    elif y_pred == 'car_horn':
        img = 'static/img/car.jpg'
        return 'Car Horn', img 
    elif y_pred == 'children_playing':
        img = 'static/img/children.jpg'
        return 'Children Playing', img
    elif y_pred == 'dog_bark':
        img = 'static/img/dog.jpg'
        return 'Dog Bark', img 
    elif y_pred == 'drilling':
        img = 'static/img/drill.jpg'
        return 'Drilling', img 
    elif y_pred == 'engine_idling':
        img = 'static/img/idling.jpg'
        return 'Engine Idling', img
    elif y_pred == 'gun_shot':
        img = 'static/img/gun.jpg'
        return 'Gun Shot', img
    elif y_pred == 'jackhammer':
        img = 'static/img/jackhammer.jpg'
        return 'Jackhammer', img
    elif y_pred == 'siren':
        img = 'static/img/siren.jpg'
        return 'Police Siren', img
    elif y_pred == 'street_music':
        img = 'static/img/street_music.jpg'
        return 'Street Music', img 


if __name__ == '__main__':
    with open(ROOT_FOLDER + 'static/model/svm.pkl', 'rb') as f1:
        svm = cPickle.load(f1)
    app.run(host='0.0.0.0', debug=True)