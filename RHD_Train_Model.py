import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import LabelEncoder


def load_data(combined_csv_path):
    """
    Load the combined feature CSV file.

    Parameters:
        combined_csv_path (str): Path to the combined CSV file.

    Returns:
        X (DataFrame): Feature matrix.
        y (Series): Labels.
    """
    print(f"Loading data from {combined_csv_path}...")
    df = pd.read_csv(combined_csv_path)
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Features are all columns starting with 'IC_'
    feature_columns = [col for col in df.columns if col.startswith('IC_')]
    X = df[feature_columns]
    y = df['label']

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    return X, y


def encode_labels(y):
    """
    Encode string labels into numerical values.

    Parameters:
        y (Series): Original labels.

    Returns:
        y_encoded (ndarray): Encoded labels.
        label_encoder (LabelEncoder): Fitted label encoder.
    """
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Classes found: {label_encoder.classes_}")
    return y_encoded, label_encoder


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (DataFrame): Feature matrix.
        y (ndarray): Encoded labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing set shape: {X_test.shape}, {y_test.shape}")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.

    Parameters:
        X_train (DataFrame): Training feature matrix.
        y_train (ndarray): Training labels.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Seed used by the random number generator.

    Returns:
        model (RandomForestClassifier): Trained Random Forest model.
    """
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest training completed.")
    return model


def evaluate_model(model, X_test, y_test, label_encoder, output_dir='outputs'):
    """
    Evaluate the trained model and generate evaluation reports and visualizations.

    Parameters:
        model (RandomForestClassifier): Trained model.
        X_test (DataFrame): Testing feature matrix.
        y_test (ndarray): Testing labels.
        label_encoder (LabelEncoder): Fitted label encoder.
        output_dir (str): Directory to save the outputs.

    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=label_encoder.transform(['RHD'])[0])
    recall = recall_score(y_test, y_pred, pos_label=label_encoder.transform(['RHD'])[0])
    f1 = f1_score(y_test, y_pred, pos_label=label_encoder.transform(['RHD'])[0])

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    print("Classification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Generate Classification Report
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("\nClassification Report:")
    print(class_report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=label_encoder.transform(['RHD'])[0])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved to {roc_path}")

    # Feature Importance
    importances = model.feature_importances_
    feature_names = X_test.columns
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances[:20], y=feature_importances.index[:20])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Feature Importances')
    fi_path = os.path.join(output_dir, 'feature_importances.png')
    plt.savefig(fi_path)
    plt.close()
    print(f"Feature importances plot saved to {fi_path}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save metrics and classification report to a text file
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)
    print(f"Evaluation report saved to {report_path}")

    return metrics


def save_model(model, label_encoder, scaler, ica, output_dir='outputs'):
    """
    Save the trained model and encoders to disk.

    Parameters:
        model (RandomForestClassifier): Trained model.
        label_encoder (LabelEncoder): Fitted label encoder.
        scaler (StandardScaler): Fitted scaler used during feature extraction.
        ica (FastICA): Fitted FastICA model used during feature extraction.
        output_dir (str): Directory to save the models.
    """
    print("Saving the trained model and encoders...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, 'random_forest_model.pkl')
    joblib.dump(model, model_path)
    print(f"Random Forest model saved to {model_path}")

    label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to {label_encoder_path}")

    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    ica_path = os.path.join(output_dir, 'fastica_model.pkl')
    joblib.dump(ica, ica_path)
    print(f"FastICA model saved to {ica_path}")


def main():
    # Configuration
    combined_csv_path = 'fastica_features_combined.csv'  # Path to the combined CSV file
    output_dir = 'outputs'  # Directory to save outputs like plots and reports
    model_output_dir = os.path.join(output_dir, 'models')  # Directory to save trained models

    # Step 1: Load Data
    X, y = load_data(combined_csv_path)

    # Step 2: Encode Labels
    y_encoded, label_encoder = encode_labels(y)

    # Step 3: Split Data
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    # Step 4: Train Random Forest
    rf_model = train_random_forest(X_train, y_train)

    # Step 5: Evaluate Model
    metrics = evaluate_model(rf_model, X_test, y_test, label_encoder, output_dir=output_dir)

    # Step 6: Save the Trained Model and Encoders
    save_model(rf_model, label_encoder, None, None, output_dir=model_output_dir)

    print("Model training and evaluation pipeline completed successfully.")


if __name__ == "__main__":
    main()
