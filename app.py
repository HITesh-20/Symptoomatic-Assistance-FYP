from flask import Flask,render_template, request,redirect,url_for
import numpy as np
import pickle
import fyp as m

app=Flask(__name__)

model = pickle.load(open("pickle.pkl", "rb"))

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/predict", methods = ["POST","GET"])
def predict():
    int_features=[int(x) for x in request.form.values()]
    arr=[np.array(int_features)]
    prediction=model.predict(arr)
    if prediction == 0:
        return render_template("index.html", pred="Less chances of having Heart Disease")
    else:
        return render_template("index.html", pred="High Chances of having heart disease")

if __name__ == "__main__":
    app.run(debug=True)



