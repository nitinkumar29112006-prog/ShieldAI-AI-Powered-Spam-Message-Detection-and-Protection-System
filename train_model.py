from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "spam_pipeline.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_FROM_REMAINDER = 0.20
MINIMUM_WORDS = 3


def find_dataset():
    """Find a likely SMS Spam Collection dataset file in the project's data folder."""
    supported = {".csv", ".tsv", ".txt", ".data", ""}
    candidates = [
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
        and not p.name.lower().startswith(("readme", "license"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset found under {DATA_DIR}.\n"
            "Place the UCI SMS Spam Collection file (often named SMSSpamCollection) "
            "inside the data folder."
        )

    # Prefer standard UCI filenames.
    preferred_names = {
        "smsspamcollection", "smsspamcollection.txt", "spam.csv",
        "sms_spam.csv", "spam.tsv"
    }
    candidates.sort(
        key=lambda p: (
            p.name.lower() not in preferred_names,
            "smsspamcollection" not in p.name.lower(),
            len(str(p)),
        )
    )
    return candidates[0]


def load_dataset(path):
    """Load common UCI SMS Spam Collection and CSV formats."""
    # UCI file is tab-separated, usually without a header: label<TAB>message.
    try:
        df = pd.read_csv(path, sep="\t", header=None, encoding="utf-8", on_bad_lines="skip")
        if df.shape[1] >= 2 and str(df.iloc[0, 0]).strip().lower() not in {
            "label", "v1", "category", "class"
        }:
            df = df.iloc[:, :2].copy()
            df.columns = ["label", "message"]
        else:
            df = None
    except Exception:
        df = None

    # If tab-separated parsing did not work, try a normal CSV with common headers.
    if df is None or not {"label", "message"}.issubset(df.columns):
        try:
            raw = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")

        lower_to_original = {str(c).strip().lower(): c for c in raw.columns}
        label_col = next(
            (lower_to_original[k] for k in ("label", "v1", "category", "class", "type")
             if k in lower_to_original),
            None,
        )
        message_col = next(
            (lower_to_original[k] for k in ("message", "v2", "text", "sms", "content")
             if k in lower_to_original),
            None,
        )

        if label_col is not None and message_col is not None:
            df = raw[[label_col, message_col]].copy()
            df.columns = ["label", "message"]
        elif raw.shape[1] >= 2:
            # Common Kaggle spam.csv format: v1 = label, v2 = message.
            df = raw.iloc[:, :2].copy()
            df.columns = ["label", "message"]
        else:
            raise ValueError(
                "Could not identify the label and message columns in the dataset. "
                "Expected label/message, v1/v2, or a two-column tab-separated file."
            )

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    label_map = {
        "spam": "spam",
        "ham": "ham",
        "1": "spam",
        "0": "ham",
        "junk": "spam",
        "not spam": "ham",
    }
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])
    df = df[df["message"].str.len() > 0]
    df = df.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both spam and ham labels. "
            f"Labels found after cleaning: {sorted(df['label'].dropna().unique().tolist())}"
        )
    if df["label"].value_counts().min() < 5:
        raise ValueError("Need at least 5 examples of each class to train and evaluate safely.")

    return df


def build_features():
    """Combine word-level and character-level TF-IDF features."""
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=100_000,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=100_000,
    )
    return FeatureUnion(
        [
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
        ],
        transformer_weights={"word_tfidf": 1.0, "char_tfidf": 0.8},
    )


def make_models():
    features = build_features()
    return {
        "Logistic Regression": Pipeline(
            [
                ("tfidf", clone(features)),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight=None,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
        "Multinomial Naive Bayes": Pipeline(
            [
                ("tfidf", clone(features)),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        ),
        # Calibrated Linear SVM provides predict_proba, so the Streamlit app can
        # show a score and apply a validation-selected decision threshold.
        "Linear SVM (calibrated)": Pipeline(
            [
                ("tfidf", clone(features)),
                ("classifier", CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight=None, random_state=RANDOM_STATE),
                    method="sigmoid",
                    cv=3,
                )),
            ]
        ),
    }


def metrics_at_threshold(y_true, spam_scores, threshold):
    pred = np.where(spam_scores >= threshold, "spam", "ham")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "spam_precision": float(precision_score(y_true, pred, pos_label="spam", zero_division=0)),
        "spam_recall": float(recall_score(y_true, pred, pos_label="spam", zero_division=0)),
        "spam_f1": float(f1_score(y_true, pred, pos_label="spam", zero_division=0)),
    }


def choose_threshold(y_true, spam_scores):
    """
    Select a threshold using validation data only.
    Prefer the best spam F1, with a light precision preference for ties.
    """
    candidates = np.arange(0.20, 0.801, 0.01)
    rows = []
    for threshold in candidates:
        row = metrics_at_threshold(y_true, spam_scores, float(threshold))
        row["threshold"] = float(threshold)
        rows.append(row)

    # Maximize F1 first; among near-ties, prefer precision.
    best = max(rows, key=lambda r: (r["spam_f1"], r["spam_precision"], r["spam_recall"]))
    return best["threshold"], best, rows


def get_spam_scores(model, texts):
    if not hasattr(model, "predict_proba"):
        raise TypeError("The selected model must expose predict_proba for threshold tuning.")
    classes = [str(c).lower() for c in model.classes_]
    if "spam" not in classes:
        raise ValueError(f"Model classes do not include 'spam': {classes}")
    spam_index = classes.index("spam")
    return model.predict_proba(texts)[:, spam_index]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset()
    df = load_dataset(dataset_path)
    X = df["message"].to_numpy()
    y = df["label"].to_numpy()

    # Hold out the final test set first. It is not used for model/threshold selection.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VALIDATION_SIZE_FROM_REMAINDER,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    print(f"Dataset: {dataset_path.relative_to(ROOT)}")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Class counts: {df['label'].value_counts().to_dict()}")
    print(f"Split sizes: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
    print("\nTraining and comparing models on the training split...")

    model_results = {}
    fitted_models = {}

    for name, model in make_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        scores = get_spam_scores(model, X_val)
        threshold, val_metrics, threshold_rows = choose_threshold(y_val, scores)
        model_results[name] = {
            "validation_metrics_at_selected_threshold": val_metrics,
            "selected_threshold": threshold,
            "threshold_search": threshold_rows,
        }
        fitted_models[name] = model
        print(
            f"{name}: validation accuracy={val_metrics['accuracy']:.3f}, "
            f"spam precision={val_metrics['spam_precision']:.3f}, "
            f"spam recall={val_metrics['spam_recall']:.3f}, "
            f"spam F1={val_metrics['spam_f1']:.3f}, "
            f"threshold={threshold:.2f}"
        )

    # Select using validation spam F1, then precision. Never select from test metrics.
    best_name = max(
        model_results,
        key=lambda name: (
            model_results[name]["validation_metrics_at_selected_threshold"]["spam_f1"],
            model_results[name]["validation_metrics_at_selected_threshold"]["spam_precision"],
            model_results[name]["validation_metrics_at_selected_threshold"]["spam_recall"],
        ),
    )
    chosen_threshold = float(model_results[best_name]["selected_threshold"])
    print(f"\nSelected model: {best_name}")
    print(f"Validation-selected spam threshold: {chosen_threshold:.3f}")

    # Refit the chosen model on training + validation data, keeping the test set untouched.
    final_model = clone(make_models()[best_name])
    X_trainval_all = np.concatenate([X_train, X_val])
    y_trainval_all = np.concatenate([y_train, y_val])
    final_model.fit(X_trainval_all, y_trainval_all)

    test_scores = get_spam_scores(final_model, X_test)
    test_metrics = metrics_at_threshold(y_test, test_scores, chosen_threshold)
    y_test_pred = np.where(test_scores >= chosen_threshold, "spam", "ham")
    cm = confusion_matrix(y_test, y_test_pred, labels=["ham", "spam"])
    report = classification_report(
        y_test, y_test_pred, labels=["ham", "spam"], output_dict=True, zero_division=0
    )

    artifact = {
        "pipeline": final_model,
        "spam_threshold": chosen_threshold,
        "minimum_words": MINIMUM_WORDS,
        "model_name": best_name,
    }
    joblib.dump(artifact, MODEL_PATH)

    output = {
        "dataset": dataset_path.name,
        "rows_after_cleaning": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "random_state": RANDOM_STATE,
        "selected_model_by_validation_spam_f1": best_name,
        "validation_metrics_by_model": {
            name: result["validation_metrics_at_selected_threshold"]
            for name, result in model_results.items()
        },
        "threshold_by_model": {
            name: float(result["selected_threshold"])
            for name, result in model_results.items()
        },
        "selected_spam_threshold": chosen_threshold,
        "test_metrics_at_selected_threshold": test_metrics,
        "test_confusion_matrix_labels_ham_spam": cm.tolist(),
        "test_classification_report": report,
        "important_caveat": (
            "Metrics are estimates on a held-out split of this SMS dataset. "
            "They do not guarantee performance on newer scams, other languages, "
            "or long-form emails. The displayed score may not be perfectly calibrated."
        ),
    }
    (REPORT_DIR / "model_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    print("\nFinal held-out test metrics:")
    print(f"Accuracy:       {test_metrics['accuracy']:.3f}")
    print(f"Spam precision: {test_metrics['spam_precision']:.3f}")
    print(f"Spam recall:    {test_metrics['spam_recall']:.3f}")
    print(f"Spam F1-score:  {test_metrics['spam_f1']:.3f}")
    print("\nConfusion matrix (rows=true, columns=predicted; order=ham, spam):")
    print(cm)
    print(f"\nSaved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved report: {(REPORT_DIR / 'model_metrics.json').relative_to(ROOT)}")
    print("\nRestart/refresh Streamlit to use the newly trained model.")


if __name__ == "__main__":
    main()
