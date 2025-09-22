import os
import time
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


model = joblib.load(r"E:\IDS\cicids_xgb_multiclass.pkl")
label_encoder = joblib.load(r"E:\IDS\label_encoder.pkl")
scaler = joblib.load(r"E:\IDS\scaler.pkl")
#category_encoders = joblib.load(r"E:\IDS\category_encoders.pkl")

WATCH_FOLDER = r"E:\IDS\incoming_flows"
LOG_FILE = r"E:\IDS\logs\prediction_log.csv"

processed_files = set()

while True:
    files = [f for f in os.listdir(WATCH_FOLDER) if f.endswith(".csv")]
    
    for file in files:
        if file in processed_files:
            continue
        
        filepath = os.path.join(WATCH_FOLDER, file)
        data = pd.read_csv(filepath)
        
        # Preprocess 
        X = data.drop(columns=["Label"], errors="ignore")
        X.columns=X.columns.str.strip()
        X = X.replace([float('inf'), -float('inf')], 0).fillna(0)
        category_encoders = joblib.load(r"E:\IDS\category_encoders.pkl")
        for col, le_col in category_encoders.items():
            if col in X.columns:
                X[col] = le_col.transform(X[col].astype(str))


        training_columns = joblib.load(r"E:\IDS\training_columns.pkl")
        X = X.reindex(columns=training_columns, fill_value=0)
        
        X_scaled = scaler.transform(X)
        
        # Prediction
        y_pred_enc = model.predict(X_scaled)
        y_pred = label_encoder.inverse_transform(y_pred_enc)
        
        # Log
        data["Predicted_Label"] = y_pred
        data.to_csv(LOG_FILE, mode="a", index=False, header=not os.path.exists(LOG_FILE))
        
        processed_files.add(file)
    
   #timer
    time.sleep(10)  
