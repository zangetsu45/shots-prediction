# xG Shot Outcome Prediction with Defensive Context and Play Patterns

This repository contains code and data pipelines for predicting shot outcomes in football (soccer) using StatsBomb open data. The model integrates spatial features, freeze-frame defensive context, and play-pattern embeddings, and is trained using an XGBoost classifier optimized with focal loss to handle class imbalance.

## 📂 Repository Structure

* `train.py` — Trains the model using SMOTE + Tomek Links and focal loss with XGBoost.
* `eval.py` — Evaluates model performance on a held-out test set with macro/micro F1, per-class recall, and confusion matrices.
* `infer.py` — Loads a trained model and runs inference on new shot data and show probability of each event       happening.
* `datasets` -
            `test.csv`
            `train.csv`
* `models/` 
#
## 🏋️ Training

To train the model from scratch using the processed dataset:

```bash
python train.py 
```

## 📊 Evaluation

To evaluate the trained model on a test set:

```bash
python eval.py 
```

Evaluation metrics include:

* Per-class precision, recall, F1-score
* Macro and weighted averages
* Confusion matrix visualization

## 🔍 Inference

To predict the outcome of new shots:

```bash
python inferernce.py 
```

This script expects an input CSV with the same schema used during training.

## 📊 Project Overview

This project enhances traditional xG models by incorporating:

* Freeze-frame defensive features (from StatsBomb 360 data)
* Play-pattern context (counterattacks, set pieces, etc.)
* Focal loss to improve prediction of rare but important classes (e.g. “Goal”, “Wayward”)

Read the full methodology and results in our report.
