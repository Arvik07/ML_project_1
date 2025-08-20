import os
from pyclbr import Class
import sys
from dataclasses import dataclass
from turtle import mode
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input features and target features") 
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            ) 
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor()
            }
            params={
                "Decision Tree":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth':[None, 10, 20, 30],
                    'splitter':['best', 'random']
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth':[None, 10, 20, 30],
                    'n_estimators':[100, 200, 300]
                },
                "Gradient Boosting":{
                    'learning_rate':[0.01, 0.1, 0.2],
                    'n_estimators':[100, 200, 300],
                    'max_depth':[None, 10, 20, 30]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[0.01, 0.1, 0.2],
                    'n_estimators':[100, 200, 300]
                },
                "K-Neighbors Regressor":{
                    'n_neighbors':[3, 5, 7, 9],
                    'weights':['uniform', 'distance']
                },
                "XGBRegressor":{
                    'learning_rate':[0.01, 0.1, 0.2],
                    'n_estimators':[100, 200, 300],
                    'max_depth':[None, 10, 20, 30]
                },
                "CatBoosting Regressor":{
                    'learning_rate':[0.01, 0.1, 0.2],
                    'n_estimators':[100, 200, 300],
                    'depth':[None, 10, 20, 30]
                },
                "Linear Regression": {}
            }

            model_report, fitted_models = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = fitted_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            return r2_score_value
        

        except Exception as e:
            logging.error("Error occurred while training model")
            raise CustomException(e, sys)