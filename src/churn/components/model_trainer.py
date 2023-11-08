import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from src.churn.logger import logging
from src.churn.exception import CustomException
from src.churn.components.data_transformation import DataTransformation
import os
import sys


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('models', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.model_config.trained_model_path), exist_ok=True)

    def train_model(self, X_train, X_test, y_train, y_test):
        try:
            #logging.info('model training started')
            #df = self.transformation_config.transformed_data_path

            #X = df.drop('Churn', axis=1)
            #y = df['Churn']
            #sm = SMOTEENN()
            #X, y = sm.fit_resample(X, y)

            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

            #print(X_train.shape, y_train.shape)

            # Random Forest Classifier
            model = RandomForestClassifier(n_estimators=100,
                                            criterion='gini',
                                            max_depth=6, 
                                            min_samples_leaf=8)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            logging.info("Random Forest Classifier Result")
            logging.info(model.score(X_test, y_test))
            logging.info(classification_report(y_test, y_pred))
            print(classification_report(y_test, y_pred))

            # Save the trained Random Forest model
            
            with open(self.model_config.trained_model_path, 'wb') as model_file:
                pickle.dump(model, model_file)

            logging.info('Trained model saved!')

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main':
    df = pd.read_csv('artifact/Final_churn_df.csv')
    trainer = ModelTrainer()
    trainer.train_model(df)
