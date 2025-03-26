import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the model, scaler, and feature names
model_path = os.path.join(PROJECT_ROOT, 'disease_prediction', 'data', 'knn_model.joblib')
scaler_path = os.path.join(PROJECT_ROOT, 'disease_prediction', 'data', 'scaler.joblib')
feature_names_path = os.path.join(PROJECT_ROOT, 'disease_prediction', 'data', 'feature_names.joblib')

print(f"Looking for model at: {model_path}")  # Debug print

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
FEATURE_NAMES = joblib.load(feature_names_path)

def home(request):
    context = {
        'FEATURE_NAMES': FEATURE_NAMES
    }
    return render(request, 'knn_app/home.html', context)

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Get symptoms from the form
            symptoms = []
            for feature in FEATURE_NAMES:
                value = request.POST.get(feature, '0')
                symptoms.append(1 if value == '1' else 0)
            
            # Create DataFrame with feature names
            input_data = pd.DataFrame([symptoms], columns=FEATURE_NAMES)
            
            # Scale the input data
            scaled_data = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(scaled_data)[0]
            probabilities = model.predict_proba(scaled_data)[0]
            
            # Get confidence score
            confidence = max(probabilities) * 100
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_predictions = [
                {
                    'disease': model.classes_[idx],
                    'probability': round(probabilities[idx] * 100, 2)
                }
                for idx in top_3_indices
            ]
            
            return JsonResponse({
                'disease': prediction,
                'confidence': round(confidence, 2),
                'top_predictions': top_3_predictions
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)
