import pandas as pd
import pickle
from src.churn.components.model_trainer import ModelTrainer


class PredictionPipeline:
    def predict_churn(self, data):
        with open('Final_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        predictions = model.predict(data)
        return predictions

    def run_prediction(self):
        df = pd.read_csv('data_to_predict.csv')
        predictor = PredictionPipeline()
        predictions = predictor.predict_churn(df)
        # You can save or use the predictions as needed

if __name__ == '__main':
    pipeline = PredictionPipeline()
    pipeline.run_prediction()
