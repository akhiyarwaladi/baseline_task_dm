"""
Utility functions for baseline KNN models
"""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


def print_header(text, level=1):
    """Print formatted section header"""
    if level == 1:
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}")
    elif level == 2:
        print(f"\n{'-'*60}")
        print(f"  {text}")
        print(f"{'-'*60}")


def evaluate_classification(model, X_train, X_test, y_train, y_test, class_names=None):
    """Evaluate classification model with standard metrics"""

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    print_header("Model Performance", level=2)
    print(f"Training Accuracy  : {train_acc:.4f}")
    print(f"Testing Accuracy   : {test_acc:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV Score (5-fold)  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Classification report
    print_header("Classification Report", level=2)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    print_header("Confusion Matrix", level=2)
    cm = confusion_matrix(y_test, y_pred)

    if class_names:
        # Print with class names
        header = "Actual \\ Predicted".ljust(20) + "  ".join([name[:8].ljust(8) for name in class_names])
        print(header)
        print("-" * len(header))
        for i, row in enumerate(cm):
            row_str = class_names[i][:18].ljust(20) + "  ".join([str(val).ljust(8) for val in row])
            print(row_str)
    else:
        print(cm)

    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'confusion_matrix': cm
    }


def print_dataset_info(df, target_col):
    """Print basic dataset information"""
    print_header("Dataset Overview", level=1)
    print(f"Total Samples  : {len(df):,}")
    print(f"Total Features : {len(df.columns)-1}")
    print(f"Target Column  : {target_col}")

    print_header("Class Distribution", level=2)
    class_counts = df[target_col].value_counts().sort_index()
    for cls, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {str(cls)[:30].ljust(30)} : {count:6,} ({pct:5.2f}%)")

    print_header("Missing Values", level=2)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values ✓")
    else:
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"  {col[:30].ljust(30)} : {count:6,} ({pct:5.2f}%)")
