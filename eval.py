import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score

# ─────────────── LOAD ARTIFACTS & DATA ───────────────
preprocessor = joblib.load("models/preprocessor.joblib")
le = joblib.load("models/label_encoder.joblib")
feat_names = joblib.load("models/feature_names.joblib")
focal_model = xgb.Booster(); focal_model.load_model("models/xgb_focal.model")

# Load saved train/test csvs
train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")

features = [col for col in train_df.columns if col != 'shot_outcome']

# ─────────────── PREPARE FUNCTION ───────────────
def prepare_data(df):
    X = df[features]
    y = df['shot_outcome']
    y_enc = le.transform(y)
    X_enc = preprocessor.transform(X)
    dmat = xgb.DMatrix(X_enc, feature_names=feat_names)
    return dmat, y_enc

# Prepare train and test
dtrain, y_train_enc = prepare_data(train_df)
dtest, y_test_enc = prepare_data(test_df)

def eval_and_plot(dmat, y_true_enc, dataset_name, model, model_name):
    # predict
    probs = model.predict(dmat)
    y_pred_enc = np.argmax(probs, axis=1)
    y_pred = le.inverse_transform(y_pred_enc)
    y_true = le.inverse_transform(y_true_enc)
    
    # metrics
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n=== {model_name} on {dataset_name} ===")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    print(classification_report(y_true, y_pred))
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{model_name} Confusion Matrix ({dataset_name})")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"images/cm_{model_name.replace(' ','_').lower()}_{dataset_name.lower()}.png")
    plt.close()

# Evaluate both models on train and test
eval_and_plot(dtrain, y_train_enc, 'Train', focal_model, 'Focal_XGB')
eval_and_plot(dtest, y_test_enc, 'Test', focal_model, 'Focal_XGB')

print("Evaluation complete. Confusion matrices saved to images/.")
