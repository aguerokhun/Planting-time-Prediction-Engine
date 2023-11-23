import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score, confusion_matrix
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from fastapi import FastAPI, HTTPException
app = FastAPI()

def train_and_save_regression_model(filename):
    df = pd.read_csv(filename)

    # Check for null values
    null_values = df.isnull().sum()
    numerical_cols = ['temperature', 'humidity', 'water availability', 'ph']
    # Fill missing values with the mean if there are
    imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Prepare the dataset for regression model
    X_reg = df[['label', 'Country']]
    X_reg_encoded = pd.get_dummies(X_reg, drop_first=True)  # Using one-hot encoding
    numerical_cols = ['temperature', 'humidity', 'water availability', 'ph']
    y_regression = df[numerical_cols]

    # Normalize numerical columns using Min-Max scaling
    scaler = MinMaxScaler()

    # Fit and transform on training data
    y_regression = scaler.fit_transform(y_regression)

    # Split the data for training and testing the regression model
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_encoded, y_regression, test_size=0.2, random_state=42)

    # Train Random Forest Regression model
    rf_regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regression_model.fit(X_train_reg, y_train_reg)

    # Save the trained model
    dump(rf_regression_model, 'regression_model.joblib')

    # Metrics for Regression Model
    y_pred_reg = rf_regression_model.predict(X_test_reg)

def train_and_save_classification_model(filename):
    df = pd.read_csv(filename)

    # Check for null values
    null_values = df.isnull().sum()
    
    numerical_cols = ['temperature', 'humidity', 'water availability', 'ph']
    
    imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    # Prepare the dataset for 'season' classification model
    
    # Normalize numerical columns using Min-Max scaling
    scaler = MinMaxScaler()

    # Fit and transform on training data
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    X_class = df[['label', 'Country'] + numerical_cols]
    X_class_encoded = pd.get_dummies(X_class, drop_first=True)  # Using one-hot encoding
    y_classification_season = df['season']
    y_classification_harvest_season = df['harvest season']
    
    # Split the data for training and testing the season classification model
    X_train_class, X_test_class, y_train_class_season, y_test_class_season = train_test_split(X_class_encoded, y_classification_season, test_size=0.2, random_state=42)
    X_train_class, X_test_class, y_train_class_harvest, y_test_class_harvest = train_test_split(X_class_encoded, y_classification_harvest_season, test_size=0.2, random_state=42)

    # Train the MultiOutputClassifier with separate Random Forest classifiers for 'season' and 'harvest season'
    multi_output_classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42), n_jobs=-1)
    multi_output_classifier.fit(X_train_class, pd.concat([y_train_class_season, y_train_class_harvest], axis=1))

    # Save the trained model
    dump(multi_output_classifier, 'classification_model.joblib')

    # Metrics for 'season' Classification Model
    y_pred_class = multi_output_classifier.predict(X_test_class)
    y_pred_class_season, y_pred_class_harvest = y_pred_class[:, 0], y_pred_class[:, 1]
    
def load_and_predict_regression_model(input_data):
    # Load the trained regression model
    loaded_regression_model = load('regression_model.joblib')

    # Preprocess input data if needed
    # ...

    # Make predictions using the loaded regression model
    predictions = loaded_regression_model.predict(input_data)

    return predictions

def load_and_predict_classification_model(input_data):
    # Load the trained classification model
    loaded_classification_model = load('classification_model.joblib')

    # Preprocess input data if needed
    # ...

    # Make predictions using the loaded classification model
    predictions = loaded_classification_model.predict(input_data)

    return predictions
#This is an example and can be uncommented for testing
# train_and_save_regression_model("C:\\Users\\aguerokhun\\Documents\\hackathon\\Crop_Data.csv")
# train_and_save_classification_model("C:\\Users\\aguerokhun\\Documents\\hackathon\\Crop_Data.csv")

# Endpoints for predictions
@app.post("/predict_regression/")
def predict_regression(input_data: str):
    try:
        # Call the function to load and predict using the regression model
        predictions = load_and_predict_regression_model(input_data)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        #Handle exceptions, return an HTTP 400 Bad Request if needed
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_classification/")
def predict_classification(input_data: str):
    predictions = load_and_predict_classification_model(input_data)
    return {"predictions": predictions.tolist()}
