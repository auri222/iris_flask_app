from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iris.pkl', 'rb'))

#Trong Procfile app:app => app đầu là tên file app.py
#                       => app kế là tên được gán cho Flask bên dưới
app = Flask(__name__)

@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    spl = request.form['sepal_length']
    spw = request.form['sepal_width']
    ptl = request.form['petal_length']
    ptw = request.form['petal_width']
    arr = np.array([[spl, spw, ptl, ptw]])
    pred = model.predict(arr)
    return render_template('result.html', data=pred)

if __name__ == '__main__' :
    app.run(debug=True)