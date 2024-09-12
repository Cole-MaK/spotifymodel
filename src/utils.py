import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.components.data_transform import DataTranformation
from sklearn.model_selection import train_test_split

import json
import base64
from requests import get, post
from dotenv import load_dotenv

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def get_feature_columns():
    raw_data = 'artifact/data.csv'
    data_transformation = DataTranformation()
    data = data_transformation.initiate_data_tranformation(raw_data)
    X = data.drop('track_genre_encoded', axis=1) 
    y = data['track_genre_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_features = X_train.columns

    return model_features

# Working with Spotify
    
load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

def get_token():
    auth_string = client_id + ':' + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes),'utf-8')

    url = 'https://accounts.spotify.com/api/token' #post request url

    headers = { # post request headers
        "Authorization": "Basic "+ auth_base64,
        "Content-Type": "application/x-www-form-urlencoded" 
    } 

    data = {'grant_type': 'client_credentials'} #post request body

    result = post(url, headers=headers, data = data)
    json_result = json.loads(result.content)
    token = json_result['access_token']
    return token

def get_auth_header(token):
    return {'Authorization': 'Bearer ' + token}