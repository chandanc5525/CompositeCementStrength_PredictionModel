# app/service.py

import os
import joblib
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


class PredictionService:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def predict(self, data):

        input_data = np.array([[
            data.cement,
            data.blast_furnace_slag,
            data.fly_ash,
            data.water,
            data.superplasticizer,
            data.coarse_aggregate,
            data.fine_aggregate,
            data.age
        ]])

        scaled_data = self.scaler.transform(input_data)
        prediction = self.model.predict(scaled_data)

        return float(prediction[0])