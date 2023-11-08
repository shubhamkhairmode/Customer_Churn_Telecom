import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.churn.logger import logging
from src.churn.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifact', 'train.csv')
    test_data_path = os.path.join('artifact', 'test.csv')
    raw_data_path = os.path.join('artifact', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

    def initiate_data_ingestion(self):
        try:
            # Load the raw data
            df = pd.read_csv(os.path.join('notebook/data/data.csv'))
            logging.info('Data loaded successfully')

            # Save the raw data to the 'raw_data_path'
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved successfully')

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=100)

            # Save the training and testing sets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion completed successfully')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
