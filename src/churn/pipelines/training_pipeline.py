# training_pipeline.py

import pandas as pd
from src.churn.components.model_trainer import ModelTrainer



class TrainingPipeline:
    def run_training(self):
        df = pd.read_csv('artifact/Final_churn_df.csv')
        trainer = ModelTrainer()
        trainer.train_model(df)

if __name__ == '__main':
    pipeline = TrainingPipeline()
    pipeline.run_training()
