from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite



### Load Models

## Malaria
#Load Model
interpreter_malaria = tflite.Interpreter(model_path='models/malaria.tflite')
interpreter_malaria.allocate_tensors()
#get input and output tensors
input_details = interpreter_malaria.get_input_details()
output_details = interpreter_malaria.get_output_details()

## Pneumonia
# Load Model
interpreter_pneumonia = tflite.Interpreter(model_path='models/pneumonia.tflite')
interpreter_pneumonia.allocate_tensors()
#get input and output tensors
input_details = interpreter_pneumonia.get_input_details()
output_details = interpreter_pneumonia.get_output_details()

## Diabetes
# Load Model
model_diabetes = pickle.load(open('models/diabetes.pkl','rb'))

## Breast Cancer
# Load Model
model_breast = pickle.load(open('models/breast_cancer.pkl','rb'))

## Heart
# Load Model
model_heart = pickle.load(open('models/heart.pkl','rb'))

## Kidney
# Load Model
model_kidney = pickle.load(open('models/kidney.pkl','rb'))

## Liver
# Load Model
model_liver = pickle.load(open('models/liver.pkl','rb'))


# Define App
app = Flask(__name__)
CORS(app)


def predict(values, dic):
    if len(values) == 8:
        values = np.asarray(values)
        return model_diabetes.predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        values = np.asarray(values)
        return model_breast.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        values = np.asarray(values)
        return model_heart.predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        values = np.asarray(values)
        return model_kidney.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        values = np.asarray(values)
        return model_liver.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    pred = 1

    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                
                if img.shape[2] == 4:
                    img = img[:, :, :3]

                img = img.reshape((1,36,36,3))
                img = img.astype(np.float32)

                #get prediction
                interpreter_malaria.set_tensor(input_details[0]['index'], img)
                interpreter_malaria.invoke()
                output_data = interpreter_malaria.get_tensor(output_details[0]['index'])
                pred = np.argmax(output_data[0])
        except Exception as ex:
            print(ex)
            message = "Please upload a valid Image"
            return render_template('malaria.html', message = message)

    return render_template('malaria_predict.html', pred = pred)

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    pred = 0

    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img.astype(np.float32)
                img = img / 255.0
                
                #get prediction
                interpreter_pneumonia.set_tensor(input_details[0]['index'], img)
                interpreter_pneumonia.invoke()
                output_data = interpreter_pneumonia.get_tensor(output_details[0]['index'])
                pred = np.argmax(output_data[0])
        except Exception as ex:
            print(ex)
            message = "Please upload a valid Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred = pred)

if __name__ == '__main__':
	app.run(debug = True)