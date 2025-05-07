import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────── DIRECTORIES ───────────────
os.makedirs("datasets", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ─────────────── 1) LOAD & CLEAN ───────────────
df = pd.read_csv("all_shots_features.csv")
df = df.dropna(subset=["shot_outcome"])
# merge tiny saved variants
df["shot_outcome"] = df["shot_outcome"].replace([
    "Saved Off Target", "Saved to Post"], "Saved")

# ─────────────── 2) FEATURES & TARGET ───────────────
features = [
    "location_x", "location_y", "shot_distance", "shot_angle",
    "num_defenders", "min_defender_dist", "defender_density_2m",
    "body_part", "shot_type", "technique", "situation", "play_pattern",
    "under_pressure"
]
X = df[features]
y = df["shot_outcome"]
# Encode labels and split
y_enc = LabelEncoder().fit_transform(y)
le = LabelEncoder().fit(y)
n_classes = len(le.classes_)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, stratify=y_enc, test_size=0.2, random_state=42
)
train_df = X_train.copy()
train_df['shot_outcome'] = le.inverse_transform(y_train)
test_df = X_test.copy()
test_df['shot_outcome'] = le.inverse_transform(y_test)
train_df.to_csv("datasets/train.csv", index=False)
test_df.to_csv("datasets/test.csv", index=False)

# ─────────────── 3) PREPROCESSING PIPELINE ───────────────
num_cols = ["location_x", "location_y", "shot_distance", "shot_angle",
            "num_defenders", "min_defender_dist", "defender_density_2m"]
cat_cols = [c for c in features if c not in num_cols]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
])

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# ─────────────── 4) HYBRID SAMPLING ───────────────
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_train_enc, y_train)

# ─────────────── 5) FOCAL LOSS & CLASS WEIGHTS ───────────────
def focal_loss_multiclass(preds, dtrain, alpha=None, gamma=2.0):
    labels = dtrain.get_label().astype(int)
    preds = preds.reshape(len(labels), -1)
    preds -= np.max(preds, axis=1, keepdims=True)
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    alpha = np.ones(probs.shape[1]) if alpha is None else (
        np.full(probs.shape[1], alpha) if np.isscalar(alpha) else alpha
    )
    y_one_hot = np.eye(probs.shape[1])[labels]
    p_t = np.sum(y_one_hot * probs, axis=1)
    grad = np.zeros_like(probs)
    hess = np.zeros_like(probs)
    for i in range(len(labels)):
        for c in range(probs.shape[1]):
            p_ic = probs[i, c]
            if c == labels[i]:
                mod = (1 - p_ic) ** gamma
                grad[i, c] = -alpha[c] * mod * (1 - p_ic)
                hess[i, c] = alpha[c] * mod * p_ic * (1 - p_ic) * (1 + gamma * (1 - p_ic))
            else:
                mod = (1 - p_t[i]) ** gamma
                grad[i, c] = alpha[labels[i]] * mod * p_ic
                hess[i, c] = alpha[labels[i]] * mod * p_ic * (1 - p_ic)
    return grad, hess

alpha_weights = (len(y_res) / (n_classes * np.bincount(y_res)))
alpha_weights /= alpha_weights.sum()

# ─────────────── 6) DMatrix FOR XGBOOST ───────────────
feat_names = list(preprocessor.get_feature_names_out())
dtrain = xgb.DMatrix(X_res, label=y_res, feature_names=feat_names)
dtest = xgb.DMatrix(X_test_enc, label=y_test, feature_names=feat_names)

# ─────────────── 7) TRAIN FOCL LOSS MODEL ───────────────
params = {
    "max_depth": 6,
    "eta": 0.1,
    "verbosity": 1,
    "num_class": n_classes,
    "objective": "multi:softprob",
    "subsample": 0.8,
    "colsample_bytree":0.8
}

model_focal = xgb.train(
    params, dtrain, num_boost_round=200,
    obj=lambda preds, dtrain: focal_loss_multiclass(preds, dtrain, alpha=alpha_weights, gamma=2.0),
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=20, verbose_eval=20
)

# ─────────────── 8) EVALUATE FOCL LOSS MODEL ───────────────
def evaluate(model, X_enc, y_true_enc, name):
    dm = xgb.DMatrix(X_enc, feature_names=feat_names)
    probs = model.predict(dm)
    preds = np.argmax(probs, axis=1)
    y_pred = le.inverse_transform(preds)
    y_true = le.inverse_transform(y_true_enc)
    print(f"\n== {name} Evaluation ==")
    print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average='macro'))
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    plt.savefig(f"images/cm_{name.replace(' ','_').lower()}.png")
    plt.close()

evaluate(model_focal, X_res, y_res, 'Train Focal')
evaluate(model_focal, X_test_enc, y_test, 'Test Focal')

# ─────────────── 9) SAVE ───────────────
joblib.dump(preprocessor, "models/preprocessor.joblib")
joblib.dump(le, "models/label_encoder.joblib")
joblib.dump(feat_names, "models/feature_names.joblib")  # save feature names for inference
model_focal.save_model("models/xgb_focal.model")
print("Focal-loss pipeline complete.")
