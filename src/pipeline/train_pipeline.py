import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, dataset):
        try:
            logging.info('splitting data into train/split')
            X = dataset.drop('track_genre_encoded', axis=1)  #track_genre_encoded will be the y
            y = dataset['track_genre_encoded']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info('splitting complete')

            logging.info('starting training')
            model_rf = RandomForestClassifier(n_estimators=95, random_state=42, verbose=True)
            model_rf.fit(X_train, y_train)
            y_pred = model_rf.predict(X_test)

            logging.info('training done')

            logging.info('got best model')
            best_model = model_rf

            logging.info('starting save obj')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy_score(y_test, predicted)
            return accuracy_score
        
        except Exception as e:
            raise CustomException(e,sys)