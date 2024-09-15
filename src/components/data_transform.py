import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# class CustomResampler(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass
#     def fit(self, data, y=None):
#         return self
    
#     def transform(self, data):
#         data_majority = data[data['explicit'] == False]
#         data_minority = data[data['explicit'] == True]

#         data_minority_upsampled = resample(data_minority, 
#                                         replace=True, 
#                                         n_samples=len(data_majority), 
#                                         random_state=42)
#         data_balanced = pd.concat([data_majority, data_minority_upsampled])
#         logging.info('Resampling for "explicit" feature completed')

#         # Resample 'time_signature'
#         data_resampled_time_signature = data_balanced.groupby('time_signature').apply(
#             lambda x: x.sample(data_balanced['time_signature'].value_counts().max(), replace=True)).reset_index(drop=True)
#         logging.info('Resampling for "time_signature" feature completed')

#         return data_resampled_time_signature
    
@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl') 

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=DataTranformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numeric_features = ['popularity', 'duration_ms','explicit','danceability',
                                'energy','loudness','speechiness','acousticness',
                                'instrumentalness','liveness','valence','tempo']
            cat_features = ['key','mode','time_signature']

            num_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )
            logging.info('numerical pipeline complete')

            cat_pipeline = Pipeline(
                steps=[
                    ('one_hot_encoder',OneHotEncoder())
                ]
            )
            logging.info('categorical pipeline complete')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numeric_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ],
                remainder='passthrough'
            )
            logging.info("created preprocessor obj")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('read in train and test data')

            preprocessing_obj = self.get_data_transformer_object()
            logging.info('got preprocessing obj')

            drop_columns = ['Unnamed: 0','track_id','artists','album_name','track_name', 'track_genre', 'track_genre_encoded']
            input_feature_train_df = train_df.drop(columns=drop_columns, axis = 1)
            target_feature_train_df = train_df['track_genre_encoded']

            input_feature_test_df = test_df.drop(columns=drop_columns, axis = 1)
            target_feature_test_df = test_df['track_genre_encoded']
            logging.info('dropped columns')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("saving preprocessing obj")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)