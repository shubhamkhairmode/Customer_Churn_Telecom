import sys
import os
import pandas as pd
from src.churn.logger import logging
from src.churn.exception import CustomException
from src.churn.components.data_ingestion import DataIngestion
from src.churn.components.data_transformation import DataTransformation
from src.churn.components.model_trainer import ModelTrainer
#from src.churn.components.model_monitering import ModelMonitoring
#from src.churn.pipelines.training_pipeline import TrainingPipeline
#from src.churn.pipelines.prediction_pipeline import PredictionPipeline

if __name__ == '__main__':
    logging.info("The Execution was started")

    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()

        # Load the raw data for transformation
        raw_data = pd.read_csv(data_ingestion.ingestion_config.raw_data_path)

        # Perform data transformation
        transformed_data = data_transformation.initiate_data_transformation(raw_data)

        # Model Training
        
        model_trainer = ModelTrainer()
        model_trainer.train_model(transformed_data)

        # Model Monitoring (Add your monitoring logic in ModelMonitoring class)

        # Training Pipeline
        #training_pipeline = TrainingPipeline()
        #training_pipeline.run_training()

        # Prediction Pipeline
        #prediction_pipeline = PredictionPipeline()
        #prediction_pipeline.run_prediction()

    except Exception as e:
        raise CustomException(e, sys)
