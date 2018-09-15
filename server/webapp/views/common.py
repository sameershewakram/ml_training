from flask import Response, request
from webapp import app
from flask_cors import CORS
from flask_autodoc import Autodoc
import pandas as pd 
import json
import numpy as np
import math
from sklearn.externals import joblib

CORS(app)
#auto = Autodoc(app)


@app.route('/')
def index_route():
    return '/'


@app.route('/predict-time', methods=['GET', 'POST'])
#@auto.doc()
def return_predicted_time():
    json_array = request.get_json()
    
    df = pd.DataFrame.from_dict(json_array)
    df['Created Time'] = pd.to_datetime(df['Created Time'])
    timstamp_into_int = df['Created Time'].values.astype(int)
    location_code = df[['Location Code', 'Pending Orders Location Wise']].values
    X = np.insert(location_code, 1,timstamp_into_int , axis=1)
    model = joblib.load('/home/expertsvision/Desktop/delivery_predict_time/Model1_NB.pkl')
    '''
    if input_items == 3:
        model, inputs = model1(json_array)
    elif input_items == 6:
        model, inputs = model2(json_array)
    elif input_items == 7:
        model, inputs = model3(json_array)
    elif input_items == 8:
        model, inputs = model4(json_array)
    else:
        print("Please lookafter your inputs")
    '''

    predict_output = model.predict(X)
    print(predict_output)
    time = pd.Series(predict_output).astype('datetime64[ns]')
    print (time)
    l1=[]
    for t in time:
        l1.append({'predict_time':str(t).split('.')[0]})
    response = Response(json.dumps(l1))
    return response

def model1(json_array):
    df = pd.DataFrame.from_dict(json_array)
    df['Created Time'] = pd.to_datetime(df['Created Time'])
    timstamp_into_int = df['Created Time'].values.astype(int)
    location_code = df[['Location Code', 'Pending Orders Location Wise']].values
    X = np.insert(location_code, 1,timstamp_into_int , axis=1)
    model = joblib.load('/home/expertsvision/Desktop/delivery_predict_time/Model1_NB.pkl')
    return model, X

def model2(json_array):
    df = pd.DataFrame.from_dict(json_array)
    print(df[['Created Time', 'Biker Assigned Time']])
    df[['Created Time','Biker Assigned Time']] = df[['Created Time',
                                                'Biker Assigned Time']].apply(pd.to_datetime)
    print (df[['Created Time','Biker Assigned Time']])
    timstamp_into_int = df[['Created Time','Biker Assigned Time']].values.astype(int)
    location_code = df[['Location Code', 'Pending Orders Location Wise', 
                                                    'Pending Order By Biker']].values
    biker_id, levels = pd.factorize(df['Biker'])
    X= np.concatenate((timstamp_into_int, location_code), axis=1)
    X = np.insert(X, 1,biker_id , axis=1)
    model = joblib.load('/home/expertsvision/Desktop/delivery_predict_time/Model2_NB.pkl')
    return model, X

def model3(json_array):
    df = pd.DataFrame.from_dict(json_array)
    df[['Created Time','Biker Assigned Time', 'Biker Accepted Time']] = df[['Created Time',
                        'Biker Assigned Time', 'Biker Accepted Time']].apply(pd.to_datetime)
    print (df[['Created Time','Biker Assigned Time', 'Biker Accepted Time']])
    timstamp_into_int = df[['Created Time','Biker Assigned Time', 'Biker Accepted Time']].values.astype(int)
    location_code = df[['Location Code', 'Pending Orders Location Wise', 'Pending Order By Biker']].values
    biker_id, levels = pd.factorize(df['Biker'])
    X= np.concatenate((timstamp_into_int, location_code), axis=1)
    X = np.insert(X, 1,biker_id , axis=1)
    model = joblib.load('/home/expertsvision/Desktop/delivery_predict_time/Model3_NB.pkl')
    return model, X

def model4(json_array):
    df = pd.DataFrame.from_dict(json_array)
    df[['Created Time','Biker Assigned Time', 'Biker Accepted Time', 'In Bike TIme']] = df[['Created Time',
                    'Biker Assigned Time', 'Biker Accepted Time', 'In Bike TIme']].apply(pd.to_datetime)
    print (df[['Created Time','Biker Assigned Time', 'Biker Accepted Time']])
    timstamp_into_int = df[['Created Time','Biker Assigned Time', 'Biker Accepted Time',
                                            'In Bike TIme']].values.astype(int)
    location_code = df[['Location Code', 'Pending Orders Location Wise', 'Pending Order By Biker']].values
    biker_id, levels = pd.factorize(df['Biker'])
    X= np.concatenate((timstamp_into_int, location_code), axis=1)
    X = np.insert(X, 1,biker_id , axis=1)
    model = joblib.load('/home/expertsvision/Desktop/delivery_predict_time/Model4_NB.pkl')
    return model, X

@app.route('/docs')
def return_api_docs():
    """
    api docs route
    :return:
    """
    return auto.html()