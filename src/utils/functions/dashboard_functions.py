import pandas as pd 
import numpy as np 
import os
import joblib
import streamlit as st
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
import gspread

def predict_umap_coordinates(user_input: pd.DataFrame, predictors: list, model_dir: str, type: str) -> tuple:
    umap_coords = []

    for i in range(1, 4):
        target_col = f"UMAP_{i}"
        
        # Paths to the necessary files
        scaler_path = os.path.join(model_dir, f"{target_col}_scaler.pkl")
        params_path = os.path.join(model_dir, f"{target_col}_corrected_params.pkl")
        
        # Load the scaler and the pre-calculated corrected parameters
        scaler = joblib.load(scaler_path)
        corrected_params = joblib.load(params_path)
        
        beta_corrected = corrected_params['beta_corrected']
        intercept_corrected = corrected_params['intercept_corrected']
        
        # Prepare the new data point
        X_new = user_input[predictors].values.reshape(1, -1)
        X_new_scaled = scaler.transform(X_new)
        
        # Apply the linear model directly with the corrected parameters
        pred_corrected = np.dot(X_new_scaled, beta_corrected) + intercept_corrected
        
        umap_coords.append(pred_corrected[0])
        
    return tuple(float(coord) for coord in umap_coords)

# COMPUTE EUCLIDEAN DISTANCE FOR EVRERY JOB IN THE DATAFRAME
def find_closest_jobs(user_coords: tuple, job_df: pd.DataFrame) -> pd.DataFrame:
    
    # Compute Euclidean distance row-wise
    coords_array = job_df[['UMAP_1', 'UMAP_2', 'UMAP_3']].values
    user_array = np.array(user_coords)

    # Vectorized distance calculation
    distances = np.linalg.norm(coords_array - user_array, axis=1)

    # Add distances to DataFrame
    job_df = job_df.copy()
    job_df['distance_to_user'] = distances

    # Sort and return top N
    closest_jobs = job_df.sort_values(by='distance_to_user')

    return closest_jobs

def normalize_series(series):
    """Normalizes a pandas Series to a 0-100 scale."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([100.0] * len(series), index=series.index)
    return 100 * (series - min_val) / (max_val - min_val)

def predict_coordinate_pc(model_coeffs, scores):
    """
    Predicts a coordinate based on a linear model's coefficients and user scores.
    """
    coordinate = model_coeffs.get('Intercept', 0)
    
    for feature, coefficient in model_coeffs.items():
        if feature != 'Intercept':
            score = scores.get(feature, 0) 
            coordinate += coefficient * score
            
    return coordinate