import pandas as pd
import numpy as np
import zipfile

import json
import os
import joblib
import gzip
from glob import glob

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, make_scorer

from sklearn.model_selection import GridSearchCV


def read_zip_data(type_of_data):
    zip_path = f"files/input/{type_of_data}_data.csv.zip"
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        file_names = zip_file.namelist()
        with zip_file.open(file_names[0]) as file:
            file_df = pd.read_csv(file)
    return file_df

def clean_data(df):
    cleaned_df = df.copy()

    cleaned_df = cleaned_df.rename(columns = {"default payment next month": "default"})
    cleaned_df = cleaned_df.drop(columns = "ID")
    cleaned_df = cleaned_df.loc[cleaned_df["MARRIAGE"] != 0]
    cleaned_df = cleaned_df.loc[cleaned_df["EDUCATION"] != 0]
    cleaned_df["EDUCATION"] = cleaned_df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    
    return cleaned_df

    
def make_pipeline_rf(categorical_features):

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"  
    )


    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("classifier", RandomForestClassifier(random_state=42)) 
        ]
    )

    return pipeline


def optimize_pipeline(pipeline, X_train, y_train):

    param_grid = {
        "classifier__n_estimators": [50, 100, 200],  
        "classifier__max_depth": [None, 10, 20],     
        "classifier__min_samples_split": [2, 5, 10], 
    }


    scorer = make_scorer(balanced_accuracy_score)


    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        scoring=scorer, 
        cv=10, 
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_
    
import shutil
def create_output_directory(output_directory):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

def save_model(path, model):
    create_output_directory("files/models/")

    with gzip.open(path, "wb") as f:
        joblib.dump(model, f)

    print(f"Model saved successfully at {path}")


def evaluate_model(model, X, y, dataset_name):

    y_pred = model.predict(X)

    metrics = {
        "dataset": dataset_name,
        "precision": precision_score(y, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, average="weighted"),
        "f1_score": f1_score(y, y_pred, average="weighted"),
    }
    
    return metrics

def compute_confusion_matrix(model, X, y, dataset_name):
    """
    Computes the confusion matrix and returns it as a dictionary.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]), 
            "predicted_1": int(cm[0, 1])
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]), 
            "predicted_1": int(cm[1, 1])
        },
    }

    return cm_dict

def run_job():

    train_data = read_zip_data("train")
    test_data = read_zip_data("test")
    train_data_clean = clean_data(train_data)
    test_data_clean = clean_data(test_data)


    X_train = train_data_clean.drop("default", axis = 1)
    X_test = test_data_clean.drop("default", axis = 1)

    y_train = train_data_clean["default"]
    y_test = test_data_clean["default"] 

    categorical_features = ["SEX","EDUCATION", "MARRIAGE"]
    rf_pipeline = make_pipeline_rf(categorical_features)

    best_model, best_params = optimize_pipeline(rf_pipeline, X_train, y_train)

    save_model(
        os.path.join("files/models/", "model.pkl.gz"),
        best_model
    )

    train_metrics = evaluate_model(best_model, X_train, y_train, "train")
    test_metrics = evaluate_model(best_model, X_test, y_test, "test")

    output_path = "files/output/metrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump([train_metrics, test_metrics], f, indent=4)

    print(f"Metrics saved successfully at {output_path}") 

    train_cm = compute_confusion_matrix(best_model, X_train, y_train, "train")
    test_cm = compute_confusion_matrix(best_model, X_test, y_test, "test")

    # Load existing metrics from metrics.json
    output_path = "files/output/metrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = []

    # Append new confusion matrices
    metrics_data.append(train_cm)
    metrics_data.append(test_cm)

    # Save updated metrics
    with open(output_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Confusion matrices saved successfully at {output_path}")



if __name__ == "__main__":
    run_job()
