from flask import *
from PIL import Image
import base64
import io
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
import urllib.request
# from keras.models import load_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import Counter

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
MODEL_UPLOAD_FOLDER = 'static/modeluploads/'
PROCESSED_UPLOAD_FOLDER = 'static/preprocessuploads/'
CROP_UPLOAD = 'static/cropimg/'
load_model_processed_densenet201_softmax = load_model('static\model_processed_densenet201201_softmax_2ndtime')

app.secret_key = 'abc'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_UPLOAD_FOLDER'] = MODEL_UPLOAD_FOLDER
app.config['PROCESSED_UPLOAD_FOLDER'] = PROCESSED_UPLOAD_FOLDER
app.config['CROP_UPLOAD'] = CROP_UPLOAD
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == "POST":
        file = request.files['imgselect']
        if file:
            if allowed_file(file.filename):
                print(request.form['imgnumber'])
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['MODEL_UPLOAD_FOLDER'], filename))
                print('upload filename '+filename)
                session['modelimagefilename'] = filename
                img = cv2.imread(f"static/modeluploads/{session['modelimagefilename']}")
                imgcopy = img.copy()
                print("Shape of the image", img.shape)
                num = int(request.form['imgnumber'])
                perwidth = img.shape[1]//num
                class_tmj = []
                for i in os.listdir('static/cropimg'):
                    os.remove(os.path.join('static/cropimg', i))

                for i in os.listdir('static/preprocessuploads'):
                    os.remove(os.path.join('static/preprocessuploads', i))
                
                for i in range(num):
                    imgcopy = img[0:img.shape[0],i*perwidth:i*perwidth+perwidth]
                    cv2.imwrite(f'static/cropimg/{i}.png', imgcopy)
                for i in range(num):
                    imgcrop = cv2.imread(f"static/cropimg/{i}.png")
                    imt_thres = cv2.cvtColor(imgcrop, cv2.COLOR_BGR2GRAY)
                    hh, ww = imt_thres.shape[:2]
                    thresh = cv2.threshold(imt_thres, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
                    th = cv2.threshold(imt_thres, thresh, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    big_contour = max(contours, key=cv2.contourArea)
                    # draw white filled contour on black background
                    result = np.zeros_like(imgcrop)
                    cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)
                    edges = cv2.Canny(result, 25, 40)
                    cv2.imwrite(f"static/preprocessuploads/{i}.png", edges)
                    edgeimg = cv2.imread(f"static/preprocessuploads/{i}.png")
                    edgeimg = cv2.resize(edgeimg, (128,128))
                    edgeimg = cv2.cvtColor(edgeimg, cv2.COLOR_BGR2RGB)
                    stats = []
                    if request.form['noise'] == 'yes':
                        stats.append(1)
                    else:
                        stats.append(0)
                    if request.form['attrition'] == 'yes':
                        stats.append(1)
                    else:
                        stats.append(0)
                    if request.form['pain'] == 'yes':
                        stats.append(1)
                    else:
                        stats.append(0)
                    if request.form['occlusion'] == 'yes':
                        stats.append(1)
                    else:
                        stats.append(0)
                    stats = np.array(stats)
                    class_tmj_number = np.argmax(load_model_processed_densenet201_softmax.predict([np.array([edgeimg]), np.array([stats])]))
                    if class_tmj_number == 0:
                        class_tmj.append("Mild")
                    elif class_tmj_number == 1:
                        class_tmj.append("Moderate")
                    elif class_tmj_number == 2:
                        class_tmj.append("Normal")
                count = Counter(class_tmj)
                classtmj = ''
                print(count['Mild'],count['Moderate'],count['Normal'])
                if count['Mild'] > count['Moderate'] and count['Mild'] > count['Normal']:
                    print("helloooo")
                    classtmj = "Mild"
                elif count['Moderate'] > count['Mild'] and count['Moderate'] > count['Normal']:
                    print('Moderate detected')
                    classtmj = "Moderate"
                elif count['Normal'] > count['Mild'] and count['Normal'] > count['Moderate']:
                    classtmj = "Normal"
                retval, buffer_img = cv2.imencode('.jpg', img)
                encoded_img_data_ori = base64.b64encode(buffer_img)
                decoded_img_data_ori = encoded_img_data_ori.decode('utf-8')
                return render_template("predict.html", ori_img = decoded_img_data_ori, class_tmj=classtmj, noise = request.form['noise'],attrition = request.form['attrition'], pain = request.form['pain'], occlusion = request.form['occlusion'])

            


    if request.method == "GET":
        return render_template("predict.html")


if __name__ == '__main__':
    app.run(debug=True)