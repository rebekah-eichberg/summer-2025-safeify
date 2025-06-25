import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(
    "ignore", message="The `cv='prefit'` option is deprecated")


def make_transformer(df, r, s, drop_first=True):
    rev_cols = [c for c in df if c.startswith("embedding_")]
    summ_cols = [c for c in df if c.startswith("embed_")]
    numeric_cols = [c for c in df if c not in rev_cols+summ_cols+["category"]]

    rev_pipe = ("drop" if r == 0 else Pipeline(
        [("scale", StandardScaler()), ("pca", PCA(n_components=r, random_state=42))]))
    sum_pipe = ("drop" if s == 0 else Pipeline(
        [("scale", StandardScaler()), ("pca", PCA(n_components=s, random_state=42))]))

    return ColumnTransformer(
        [('num', StandardScaler(), numeric_cols),
         ('cat', OneHotEncoder(handle_unknown="ignore",
          drop="first" if drop_first else None, sparse_output=False), ["category"]),
         ('rev', rev_pipe,  rev_cols),
         ('sum', sum_pipe,  summ_cols)
         ]).set_output(transform="pandas")


def plot_calibration_and_error_distributions_before_true_cal(y_true, models, X, threshold=0.5, n_bins=10):
    """
    Plots calibration curve and predicted probability histograms with FP/FN overlays.

    Parameters:
    - y_true: array-like, true binary labels (0 or 1)
    - models: a single model or list/tuple of 3 models with .predict_proba()
    - X: input feature matrix
    - threshold: threshold to convert predicted probs to class labels
    - n_bins: number of bins for calibration curve and histograms
    """

    # Handle either a single model or an ensemble of 3
    if isinstance(models, (list, tuple)):
        if len(models) != 3:
            raise ValueError(
                "If passing multiple models, please provide exactly 3.")
        proba_list = [model.predict_proba(X) for model in models]
        avg_proba = sum(proba_list) / len(proba_list)
    else:
        avg_proba = models.predict_proba(X)

    probs = avg_proba[:, 1]

    # Calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, probs, n_bins=n_bins, strategy='uniform')

    # Predicted labels
    y_pred = (probs >= threshold).astype(int)

    # Masks
    true_0 = y_true == 0
    true_1 = y_true == 1
    fp_mask = (y_pred == 1) & true_0
    tn_mask = (y_pred == 0) & true_0
    tp_mask = (y_pred == 1) & true_1
    fn_mask = (y_pred == 0) & true_1

    # Subsets
    probs_fp = probs[fp_mask]
    probs_tn = probs[tn_mask]
    probs_tp = probs[tp_mask]
    probs_fn = probs[fn_mask]
    probs_true0 = probs[true_0]
    probs_true1 = probs[true_1]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Calibration Curve ###
    axs[0].plot(prob_pred, prob_true, marker='o',
                label='Calibrated Model', linewidth=2)
    axs[0].plot([0, 1], [0, 1], linestyle='--',
                color='gray', label='Perfect Calibration')
    axs[0].set_xlabel('Mean Predicted Probability (per bin)')
    axs[0].set_ylabel('True Fraction of Positives')
    axs[0].set_title('Reliability Curve')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Histogram for True Label = 0 ###
    bins = np.linspace(0, 1, n_bins + 1)
    axs[1].hist(probs_true0, bins=bins, color='lightgray',
                edgecolor='black', label='All Class 0')
    axs[1].hist(probs_fp, bins=bins, color='salmon',
                edgecolor='black', label='False Positives')
    axs[1].set_title('Samples with True Label = 0')
    axs[1].set_xlabel('Predicted Probability (Class 1)')
    axs[1].set_ylabel('Sample Count')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.5)

    ### Plot 3: Histogram for True Label = 1 ###
    axs[2].hist(probs_true1, bins=bins, color='lightgray',
                edgecolor='black', label='All Class 1')
    axs[2].hist(probs_tp, bins=bins, color='mediumseagreen',
                edgecolor='black', label='True Positives')
    axs[2].hist(probs_fn, bins=bins, color='mediumpurple',
                edgecolor='black', label='False Negatives')
    axs[2].set_title('Samples with True Label = 1')
    axs[2].set_xlabel('Predicted Probability (Class 1)')
    axs[2].set_ylabel('Sample Count')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_calibration_and_class_distributions(models, X, y_true, n_bins=10, threshold=None):
    if threshold is None:
        threshold = 1 / 146

    if isinstance(models, (list, tuple)):
        if len(models) != 3:
            raise ValueError(
                "If passing multiple models, please provide exactly 3.")
        proba_list = [model.predict_proba(X) for model in models]
        avg_proba = sum(proba_list) / len(proba_list)
    else:
        avg_proba = models.predict_proba(X)

    probs = avg_proba[:, 1]

    # Use calibration_curve to get prob_true, prob_pred and bin edges
    prob_true, prob_pred = calibration_curve(
        y_true, probs, n_bins=n_bins, strategy='uniform')

    # Recompute bin edges for uniform bins to count samples per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts, _ = np.histogram(probs, bins=bin_edges)
    non_empty_bins = bin_counts > 0

    # Compute ECE only on non-empty bins:
    ece = np.sum(np.abs(prob_true - prob_pred) *
                 bin_counts[non_empty_bins] / np.sum(bin_counts[non_empty_bins]))

    # Split by class
    probs_class0 = probs[y_true == 0]
    probs_class1 = probs[y_true == 1]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={
                            'width_ratios': [2, 1, 1]})
    bins = bin_edges

    axs[0].plot(prob_pred, prob_true, marker='o',
                label='Calibrated Model', linewidth=2)
    axs[0].plot([0, 1], [0, 1], linestyle='--',
                color='gray', label='Perfect Calibration')
    axs[0].set_xlabel('Mean Predicted Probability (per bin)')
    axs[0].set_ylabel('True Fraction of Positives')
    axs[0].set_title('Reliability Curve')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    axs[1].hist(probs_class0, bins=bins, color='lightblue',
                edgecolor='black', alpha=0.8)
    axs[1].set_title('Predicted Probabilities\n(True Label = 0)')
    axs[1].set_xlabel('Predicted Probability for Class 1')
    axs[1].set_ylabel('Sample Count')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    axs[2].hist(probs_class1, bins=bins, color='orange',
                edgecolor='black', alpha=0.8)
    axs[2].set_title('Predicted Probabilities\n(True Label = 1)')
    axs[2].set_xlabel('Predicted Probability for Class 1')
    axs[2].set_ylabel('Sample Count')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    preds = (probs >= threshold).astype(int)

    print(f"\nClassification Report @ Threshold = {threshold:.5f}")
    print(classification_report(y_true, preds))

    print(f"Calibration Error (ECE, {n_bins} bins): {ece:.5f}")


def plot_high_res_reliability_and_hist(models, X, y_true, threshold=1/146, n_bins=100, log_scale=False):
    """
    Plot high-resolution reliability curve + histogram of predicted probabilities,
    with vertical threshold line.

    Parameters:
    - models: single model or list/tuple of 3 models with .predict_proba()
    - X: features for prediction
    - y_true: true labels
    - threshold: probability threshold to show (default 1/146 ≈ 0.00685)
    - n_bins: number of bins for calibration curve and histogram (default 100)
    - log_scale: bool, if True plots use log scale on x-axis (default False)
    """

    # Compute predicted probabilities for class 1
    if isinstance(models, (list, tuple)):
        if len(models) != 3:
            raise ValueError(
                "If passing multiple models, please provide exactly 3.")
        proba_list = [model.predict_proba(X) for model in models]
        avg_proba = sum(proba_list) / len(proba_list)
    else:
        avg_proba = models.predict_proba(X)

    proba_class1 = avg_proba[:, 1]

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, proba_class1, n_bins=n_bins, strategy='uniform')

    fig, axs = plt.subplots(1, 2, figsize=(
        14, 5), gridspec_kw={'width_ratios': [2, 1]})

    # Plot 1: Reliability Curve
    axs[0].plot(prob_pred, prob_true, marker='o',
                label='Calibrated Model', linewidth=2)
    axs[0].plot([0, 1], [0, 1], linestyle='--',
                color='gray', label='Perfect Calibration')

    if log_scale:
        axs[0].plot([1e-5, 1], [1e-5, 1], linestyle='--',
                    color='gray', label='Perfect Calibration')
        axs[0].set_xscale('log')
        axs[0].set_xlim(1e-4, 1)
    else:
        axs[0].set_xlim(0, 0.05)

    axs[0].set_ylim(0, 1.05)
    axs[0].set_xlabel('Mean Predicted Probability' +
                      (' (log scale)' if log_scale else ''))
    axs[0].set_ylabel('True Fraction of Positives')
    axs[0].set_title('Reliability Curve' +
                     (' (Log Scale)' if log_scale else ' (High Resolution)'))

    # Threshold line and annotation
    axs[0].axvline(x=threshold, color='red', linestyle='--')
    xpos = threshold * 1.5 if log_scale else threshold + 0.002
    axs[0].text(
        xpos, 0.1, f'Threshold ≈ {threshold:.5f}', color='red', rotation=90, va='bottom')

    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6,
                which='both' if log_scale else 'major')

    # Plot 2: Histogram of Predicted Probabilities
    if log_scale:
        bins = np.logspace(-4, 0, n_bins)
        axs[1].hist(proba_class1, bins=bins,
                    color='steelblue', edgecolor='black')
        axs[1].set_xscale('log')
        axs[1].set_xlim(1e-4, 1)
    else:
        bins = np.linspace(0, 1, n_bins)
        axs[1].hist(proba_class1, bins=bins,
                    color='steelblue', edgecolor='black')
        axs[1].set_xlim(0, 0.05)

    axs[1].axvline(x=threshold, color='red', linestyle='--',
                   label=f'Threshold ≈ {threshold:.5f}')
    axs[1].set_xlabel('Predicted Probability' +
                      (' (log scale)' if log_scale else ''))
    axs[1].set_ylabel('Sample Count')
    axs[1].set_title('Distribution of Predicted Probabilities')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.5,
                which='both' if log_scale else 'major')

    plt.tight_layout()
    plt.show()


def split_features_target(df, target='match'):
    return df.drop(columns=[target]), df[target]
