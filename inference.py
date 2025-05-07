import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
preprocessor = joblib.load("models/preprocessor.joblib")
le = joblib.load("models/label_encoder.joblib")
feat_names = joblib.load("models/feature_names.joblib")
focal_model = xgb.Booster()
focal_model.load_model("models/xgb_focal.model")

# Prepare new data
new_shots = pd.DataFrame([
    {
        "location_x": 100, "location_y": 20, "shot_distance": 18.5, "shot_angle": 0.40,
        "num_defenders": 3, "min_defender_dist": 2.0,
        "defender_density_2m": 0.8, "body_part": "head", "shot_type": "open_play",
        "technique": "normal", "situation": "lawda", "play_pattern": "possession",
        "under_pressure": False
    }
])

# Transform and predict
X_proc = preprocessor.transform(new_shots)
# build DMatrix with names
dmat = xgb.DMatrix(X_proc, feature_names=feat_names)
probs_focal = focal_model.predict(dmat)[0]   # shape (n_classes,)

print("\nShot Prediction Model:")
for cls, p in zip(le.classes_, probs_focal):
    print(f"{cls}: {p*100:.2f}%")
labels = [f"{cls}: {p*100:.2f}%" for cls, p in zip(le.classes_, probs_focal)]
sizes  = probs_focal  # already sums to 1

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, startangle=90)
plt.title("Shot Outcome Probabilities")
plt.tight_layout()
plt.show()