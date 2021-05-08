import numpy as np
import pandas as pd 
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
s_scaler = preprocessing.StandardScaler()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    stan_data = pd.read_csv('stan.csv')
    stan_data=stan_data.drop(["fetal_health"],axis=1)
    stan_arr = np.array(stan_data)
    stan_arr = np.append(stan_arr, np.array(final_features), axis=0)
    stan_arr= s_scaler.fit_transform(stan_arr)
    main_pred = stan_arr[2126]
    # main_pred
    prediction = model.predict([main_pred])
    print(prediction)

    # prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text=format(output))
    


if __name__ == "__main__":
    app.run(debug=True)