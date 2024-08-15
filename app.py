import pickle
from flask import Flask,app,request,jsonify,render_template,url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

#load the model
model = pickle.load(open("regression.pkl",'rb'))
scaler = pickle.load(open("scalling.pkl","rb"))

@app.route('/')
def home():
    return render_template("Home.html")

@app.route('/predict.api',methods =['POST'])
def predict_api():
    data = request.json['Data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.tranform(np.array(list(data.values())).reshape(1,-1))
    answer = model.predict(new_data)
    return jsonify(answer[0])

if __name__ == "__main__":
    app.run(debug=True)


    




