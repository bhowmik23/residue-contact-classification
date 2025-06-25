
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ubjson
import xgboost as xgb


def train_ovr_model(class_idx, X_train, y_train, X_val, y_val, params, num_boost_round=2000):
    """Train a one-vs-rest XGBoost model for class `class_idx` using xgb.train()"""
    # Create binary labels for this class
    y_train_bin = (y_train == class_idx).astype(int)
    y_val_bin = (y_val == class_idx).astype(int)

    # Wrap data into DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train_bin, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val_bin, enable_categorical=True)

    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=15,
        verbose_eval=50,
        evals_result=evals_result
    )

    return booster


def train_ovr_ensemble(model_name, X_train, y_train, X_val, y_val, base_params, num_classes=8):
    models = []
    for class_idx in range(num_classes):
        print(f"--- Training model {model_name} for class {class_idx} ---")
        booster = train_ovr_model(
            class_idx, X_train, y_train, X_val, y_val, base_params
        )
        models.append(booster)
    return models


def save_ovr_models(models, model_name, model_dir):
    for i, booster in enumerate(models):
        filename = f"{model_dir}/{model_name}_class_{i}.ubj"
        booster.save_model(filename)
        print(f"Saved model for class {i} → {filename}")


def predict_ovr(models, X_test):
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    probs = np.zeros((X_test.shape[0], len(models)))

    for class_idx, booster in enumerate(models):
        probs[:, class_idx] = booster.predict(dtest)

    preds = np.argmax(probs, axis=1)
    return preds, probs


def load_ovr_models(model_name, model_dir, model_parameters, num_classes=8):
    models = []
    for i in range(num_classes):
        filename = f"{model_dir}/{model_name}_class_{i}.ubj"
        booster = xgb.Booster()
        booster.load_model(filename)
        models.append(booster)

    for model in models:
        model.set_param(model_parameters)

    return models


def plot_confusion_matrix(cm, labels, model_name="Model"):
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='viridis', cbar=True,
                xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {model_name} (row-normalized colors)')
    plt.tight_layout()
    plt.show()
