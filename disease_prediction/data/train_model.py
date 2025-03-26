import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Define feature names
FEATURE_NAMES = [
    'Fever', 'Cough', 'Fatigue', 'Body Pain', 'Headache', 'Sore Throat',
    'Nausea', 'Vomiting', 'Diarrhea', 'Chest Pain', 'Shortness of Breath',
    'Loss of Taste', 'Loss of Smell', 'Skin Rash', 'Joint Pain', 'Muscle Pain', 'Chills'
]

# Load and prepare data
data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
df = pd.read_csv(data_path)

X = df[FEATURE_NAMES]
y = df['Disease']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train_scaled, y_train)

# Save files
current_dir = os.path.dirname(__file__)
joblib.dump(knn, os.path.join(current_dir, 'knn_model.joblib'))
joblib.dump(scaler, os.path.join(current_dir, 'scaler.joblib'))
joblib.dump(FEATURE_NAMES, os.path.join(current_dir, 'feature_names.joblib'))

# Print results
print("\nTraining accuracy:", knn.score(X_train_scaled, y_train))
print("Testing accuracy:", knn.score(X_test_scaled, y_test))
print("\nClass distribution:")
print(y.value_counts())
