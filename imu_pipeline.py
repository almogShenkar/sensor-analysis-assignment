# imu_pipeline.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Define signal feature columns
signal_features = [
    'x_mean', 'x_std', 'x_max', 'x_min', 'x_range', 'x_skew', 'x_kurtosis',
    'x_n_peaks', 'x_energy',
    'y_mean', 'y_std', 'y_max', 'y_min', 'y_range', 'y_skew', 'y_kurtosis',
    'y_n_peaks', 'y_energy',
    'z_mean', 'z_std', 'z_max', 'z_min', 'z_range', 'z_skew', 'z_kurtosis',
    'z_n_peaks', 'z_energy',
    'mag_mean', 'mag_std', 'mag_max', 'max_delta_mag', 'sudden_change_score',
    'x_fft_max', 'x_fft_mean',
    'y_fft_max', 'y_fft_mean',
    'z_fft_max', 'z_fft_mean',
    'xy_corr', 'xz_corr', 'yz_corr'
]


class IMUPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        y_enc = self.label_encoder.fit_transform(y)
        self.model.fit(X[signal_features], y_enc)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X[signal_features])
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        return self.model.predict_proba(X[signal_features])
