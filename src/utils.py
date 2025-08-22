import os
import sys
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    try:
        report = {}
        fitted_models = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param_grid = params[model_name]
            gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
            fitted_models[model_name] = best_model
        return report, fitted_models
    except Exception as e:
        logging.error("Error occurred while evaluating models")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        # Ensure file_path is absolute; if not, resolve relative to this file's directory
        if not os.path.isabs(file_path):
            base_dir = os.path.dirname(__file__)
            file_path = os.path.abspath(os.path.join(base_dir, file_path))
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.error(f"Error occurred while loading object from {file_path}")
        raise CustomException(e, sys)