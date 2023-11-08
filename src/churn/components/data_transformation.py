import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.churn.logger import logging
from src.churn.exception import CustomException
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

@dataclass
class DataTransformationConfig:
    transformed_data_path = os.path.join('artifact', 'Final_churn_df.csv')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        os.makedirs(os.path.dirname(self.transformation_config.transformed_data_path), exist_ok=True)

    def initiate_data_transformation(self, df):
        try:
            logging.info("Data transformation started")

            # Drop 'customerID' column
            df.drop('customerID', axis=1, inplace=True)
            logging.info("Successfully dropped 'customerID' column")

            # Drop rows with any missing values
            df.dropna(how='any', inplace=True)
            logging.info("Successfully removed rows with missing values")

            # Group 'tenure' into bins of 12 months
            labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
            df['tenure_group'] = pd.cut(df['tenure'], range(1, 80, 12), right=False, labels=labels)
            logging.info("Successfully created 'tenure_group' column")

            # Drop the old 'tenure' column
            df.drop('tenure', axis=1, inplace=True)
            logging.info("Successfully dropped the old 'tenure' column")

            # Encode 'Churn' to 1 for 'Yes' and 0 for 'No'
            df['Churn'] = (df['Churn'] == 'Yes').astype(int)
            logging.info("Successfully encoded 'Churn' feature")

            # Label encode categorical columns
            label_encoder = LabelEncoder()
            categorical_columns_to_encode = [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod', 'tenure_group'
            ]

            for column in categorical_columns_to_encode:
                df[column] = label_encoder.fit_transform(df[column])
            logging.info("Successfully encoded categorical columns")

            # Save the transformed DataFrame to a CSV file
            df.to_csv(self.transformation_config.transformed_data_path, index=False)
            logging.info("Successfully saved the transformed DataFrame to CSV")

            logging.info("Data transformation completed")

            X = df.drop('Churn', axis=1)
            y = df['Churn']

            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=100)

            logging.info(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

            return(
                X_train,
                X_test,
                y_train,
                y_test
            )


            

        except Exception as e:
            raise CustomException(e, sys)
