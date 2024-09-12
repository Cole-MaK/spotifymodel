import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import resample

from src.exception import CustomException
from src.logger import logging


class DataTranformation:
    def __init__(self):
        pass
    
    def initiate_data_tranformation(self, raw_path):
        try:
            data=pd.read_csv(raw_path)

            logging.info("Read train and test data completed")
            logging.info('initiating data preprocessing')

            data = data.dropna()

            label_encoder = LabelEncoder()
            data['track_genre_encoded'] = label_encoder.fit_transform(data['track_genre'])

            df_majority = data[data['explicit'] == False]
            df_minority = data[data['explicit'] == True]

            df_minority_upsampled = resample(df_minority, 
                                            replace=True,
                                            n_samples=len(df_majority),
                                            random_state=42)

            df_balanced = pd.concat([df_majority, df_minority_upsampled])

            data_resampled_time_signature = df_balanced.groupby('time_signature').apply(
                lambda x: x.sample(df_balanced['time_signature'].value_counts().max(), replace=True)).reset_index(drop=True)
            logging.info('resampling complete')

            data_resampled = data_resampled_time_signature.drop(['Unnamed: 0','track_id','album_name','track_name','track_genre', 'artists'],axis =1)

            scaler = StandardScaler()
            continuous_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 
                        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
            data_resampled[continuous_features] = scaler.fit_transform(data_resampled[continuous_features])

            data_resampled = pd.get_dummies(data_resampled, columns=['key', 'mode', 'time_signature'], drop_first=True)

            logging.info('preprocessing finished')

            return data_resampled
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def transform_input_data(self, dataframe):
        try:
            scaler = StandardScaler()
            continuous_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 
                        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
            dataframe[continuous_features] = scaler.fit_transform(dataframe[continuous_features])
            
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
