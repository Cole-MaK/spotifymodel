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
    
    def initiate_model_trainer(self, train_array, test_array):
        try:

            logging.info('splitting data into train/split')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('splitting complete')

            models = {
                "Random Forest": RandomForestClassifier(n_estimators=95, random_state=42, verbose=True)
            }

            logging.info('starting training')
            print('starting training')
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models)
            logging.info('training done')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info('got best model')

            if best_model_score < 0.6:
                raise CustomException("No Best Model found")
            logging.info(f'Best model found on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('saved model')
            
            predicted=best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)
            return acc_score
            # model_rf = RandomForestClassifier(n_estimators=95, random_state=42, verbose=True)
            # model_rf.fit(X_train, y_train)

            # y_pred = model_rf.predict(X_test)


            # logging.info('got best model')
            # best_model = model_rf

            # logging.info('starting save obj')

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj=best_model
            # )

            # predicted=best_model.predict(X_test)

            # acc_score = accuracy_score(y_test, predicted)
            # print(acc_score)
            # return acc_score
    
        except Exception as e:
            raise CustomException(e,sys)
