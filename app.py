# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:25:43 2023

@author: didri
"""

import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask('__name__')

model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if prediction[0] == 1:
        output = "The patient has Parkinson's"
    elif prediction[0] == 0:
        output = "The patient does not have Parkinson's"
    else:
        output = "Unexpected prediction value"
    
    return render_template("index.html", prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
