import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load & clean
df = pd.read_csv("all_shots_features.csv")
df = df.dropna(subset=["shot_outcome"])

# 2) (Optional) Merge saved variants
df['shot_outcome'] = df['shot_outcome'].replace(
    ['Saved Off Target', 'Saved to Post'], 'Saved'
)

# 3) Define features & target
features = [
    'location_x','location_y','shot_distance','shot_angle',
    'num_defenders','min_defender_dist','defender_density_2m',
    'body_part','shot_type','technique','situation','play_pattern',
    'under_pressure'
]
X = df[features]
y = df['shot_outcome']

# 4) Encode target labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 5) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, stratify=y_enc, test_size=0.2, random_state=42
)

# 6) Preprocessing pipelines
num_cols = [
    'location_x','location_y','shot_distance','shot_angle',
    'num_defenders','min_defender_dist','defender_density_2m'
]
cat_cols = [c for c in features if c not in num_cols]

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# 7) Preprocess train & test
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

# 8) Hybrid sampling: SMOTE + Tomek links
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_train_enc, y_train)
print("Resampled class distribution:", Counter(y_res))

# 9) Train XGBoost on resampled data
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_res, y_res)

# 10) Evaluate on test set
y_pred = model.predict(X_test_enc)
y_test_lbl = le.inverse_transform(y_test)
y_pred_lbl = le.inverse_transform(y_pred)

print("Balanced Accuracy:", balanced_accuracy_score(y_test_lbl, y_pred_lbl))
print("\nClassification Report:\n", classification_report(y_test_lbl, y_pred_lbl))

cm = confusion_matrix(y_test_lbl, y_pred_lbl, labels=le.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix: XGBoost + SMOTETomek")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
