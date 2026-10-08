"""
utils/model_registry.py
───────────────────────
Central Model Registry for Voice Language Identification.
Defines, instantiates, and manages all 6 classification models:
  1. Random Forest (RandomForestClassifier)
  2. Support Vector Machine (SVC)
  3. K-Nearest Neighbors (KNeighborsClassifier)
  4. Logistic Regression (LogisticRegression)
  5. Decision Tree (DecisionTreeClassifier)
  6. Gradient Boosting (GradientBoostingClassifier)

All models are trained on the same 80-dimensional MFCC feature vectors
and evaluated on the exact same train/test split.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Canonical model specifications
MODEL_DEFINITIONS = {
    "random_forest": {
        "id": "random_forest",
        "name": "Random Forest",
        "description": "Ensemble of 100 decision trees (robust baseline)",
        "filename": "random_forest.pkl",
        "legacy_filename": "language_model.pkl",
        "create": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    },
    "svm": {
        "id": "svm",
        "name": "Support Vector Machine (SVM)",
        "short_name": "SVM",
        "description": "RBF kernel Support Vector Classifier with probability estimates",
        "filename": "svm.pkl",
        "create": lambda: SVC(
            kernel="rbf", probability=True, random_state=42
        ),
    },
    "knn": {
        "id": "knn",
        "name": "K-Nearest Neighbors (KNN)",
        "short_name": "KNN",
        "description": "Non-parametric instance-based classifier (k=5)",
        "filename": "knn.pkl",
        "create": lambda: KNeighborsClassifier(
            n_neighbors=5
        ),
    },
    "logistic_regression": {
        "id": "logistic_regression",
        "name": "Logistic Regression",
        "short_name": "Logistic Regression",
        "description": "Linear multinomial classification with L2 regularization",
        "filename": "logistic_regression.pkl",
        "create": lambda: LogisticRegression(
            max_iter=1000, random_state=42
        ),
    },
    "decision_tree": {
        "id": "decision_tree",
        "name": "Decision Tree",
        "short_name": "Decision Tree",
        "description": "Single interpretable decision tree with Gini impurity",
        "filename": "decision_tree.pkl",
        "create": lambda: DecisionTreeClassifier(
            random_state=42
        ),
    },
    "gradient_boosting": {
        "id": "gradient_boosting",
        "name": "Gradient Boosting",
        "short_name": "Gradient Boosting",
        "description": "Sequential boosting ensemble of decision trees",
        "filename": "gradient_boosting.pkl",
        "create": lambda: GradientBoostingClassifier(
            random_state=42
        ),
    },
}

# In-memory cache for loaded model objects to avoid disk overhead on repeated predictions
_MODEL_CACHE = {}


def get_model_definitions():
    """Return dictionary of all supported model metadata."""
    return MODEL_DEFINITIONS


def get_model_choices():
    """
    Return ordered list of (id, display_name) for dropdowns and CLI choices.
    Includes 'auto' (Auto - Best Model).
    """
    choices = [
        ("auto", "Auto – Best Model (Highest Accuracy)"),
    ]
    for m_id, m_info in MODEL_DEFINITIONS.items():
        choices.append((m_id, m_info["name"]))
    return choices


def normalize_model_key(key: str) -> str:
    """Normalize user input or API parameter to a canonical model key."""
    if not key:
        return "random_forest"
    k = key.strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "auto": "auto",
        "best": "auto",
        "best_model": "auto",
        "rf": "random_forest",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "svm": "svm",
        "support_vector_machine": "svm",
        "svc": "svm",
        "knn": "knn",
        "k_nearest_neighbors": "knn",
        "kneighbors": "knn",
        "lr": "logistic_regression",
        "logistic": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "dt": "decision_tree",
        "tree": "decision_tree",
        "decision_tree": "decision_tree",
        "gb": "gradient_boosting",
        "gbc": "gradient_boosting",
        "gradient_boost": "gradient_boosting",
        "gradient_boosting": "gradient_boosting",
    }
    return alias_map.get(k, k if k in MODEL_DEFINITIONS else "random_forest")


def get_model_path(model_key: str) -> str:
    """Return the absolute file path for a model pkl file."""
    norm_key = normalize_model_key(model_key)
    if norm_key in MODEL_DEFINITIONS:
        fname = MODEL_DEFINITIONS[norm_key]["filename"]
        return os.path.join(MODELS_DIR, fname)
    return os.path.join(MODELS_DIR, f"{norm_key}.pkl")


def load_classifier(model_key: str, use_cache: bool = True):
    """
    Load a trained classifier from disk by model key.
    Falls back to legacy language_model.pkl if random_forest.pkl is not yet present.
    """
    norm_key = normalize_model_key(model_key)
    if norm_key not in MODEL_DEFINITIONS:
        raise ValueError(
            f"Unknown model key '{model_key}'. Supported: {list(MODEL_DEFINITIONS.keys())}"
        )

    if use_cache and norm_key in _MODEL_CACHE:
        return _MODEL_CACHE[norm_key]

    target_path = get_model_path(norm_key)

    # Backward compatibility fallback for random forest
    if not os.path.exists(target_path) and norm_key == "random_forest":
        legacy_path = os.path.join(MODELS_DIR, "language_model.pkl")
        if os.path.exists(legacy_path):
            target_path = legacy_path

    if not os.path.exists(target_path):
        m_name = MODEL_DEFINITIONS[norm_key]["name"]
        raise FileNotFoundError(
            f"The '{m_name}' model is not trained yet (missing {os.path.basename(target_path)}). "
            f"Please go to the 'Train Local Model' tab or run 'python train_model.py'."
        )

    clf = joblib.load(target_path)
    if use_cache:
        _MODEL_CACHE[norm_key] = clf
    return clf


def clear_model_cache():
    """Clear in-memory cached model instances (useful after retraining)."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
