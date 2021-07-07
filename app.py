import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Create Flask App
app = Flask(__name__)

#Load pickle model
model = pickle.load(open("model.pkl", "rb"))

#Define the homepage
@app.route("/")

def Home():
    return render_template("index.html")

#Defining predictive method
@app.route("/predict", methods = ["POST"])

def predict():
    
    #converting values into float, and saving them inside a variable
    float_features = [float(x) for x in request.form.values()]
    
    #Converting the above float values into numpy arrays
    features = [np.array(float_features)]
    
    #Making predicting calling the model
    prediction = model.predict(features)
    
    return render_template("index.html", prediction_text = "The flower's species is {}".format(prediction))

if __name__ == "__main__":
    app.run(port=5000, debug=True)