import os
import sys
from src.logger import logging
from src.exception import CustomException


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from dataclasses import dataclass

from src.components.data_transform import DataTranformation, DataTranformationConfig
from src.pipeline.train_pipeline import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifact','data.csv')
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        

    def initiate_data_ingestion(self):
        logging.info("Entered the data Ingestion method or component")
        try:
            # read the dataset from anywhere
            df = pd.read_csv('notebook/spotify.csv')

            label_encoder = LabelEncoder()
            df['track_genre_encoded'] = label_encoder.fit_transform(df['track_genre'])
            logging.info("Read dataset as df and encoded outcome")

            data_majority = df[df['explicit'] == False]
            data_minority = df[df['explicit'] == True]

            data_minority_upsampled = resample(data_minority, 
                                            replace=True, 
                                            n_samples=len(data_majority), 
                                            random_state=42)
            data_balanced = pd.concat([data_majority, data_minority_upsampled])
            logging.info('Resampling for "explicit" feature completed')

            # Resample 'time_signature'
            df = data_balanced.groupby('time_signature', group_keys=False).apply(
                lambda x: x.sample(data_balanced['time_signature'].value_counts().max(), replace=True)).reset_index(drop=True)
            logging.info('Resampling for "time_signature" feature completed')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            logging.info("Made artifact dir")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("made data.csv")

            logging.info("initiating train test split")
            train_set, test_set = train_test_split(df, test_size = .2, random_state=420)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Train test split complete | files made')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTranformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_tranformation(train_data, test_data)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))