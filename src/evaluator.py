
# evaluator.py
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight   
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(clf, X_val, y_val):
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)[:, 1]

    # Compute metrics
    recall = recall_score(y_val, y_pred, average='macro')
    f1     = f1_score(y_val, y_pred, average='macro')
    pr_auc = average_precision_score(y_val, y_pred_proba)

    print(" Validation Results:")
    print(f"Recall (macro):     {recall:.4f}")
    print(f"F1-score (macro):   {f1:.4f}")
    print(f"PR-AUC:             {pr_auc:.4f}")
    print(classification_report(y_val, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Plot Precision-Recall Curve with balanced weights
    w = compute_sample_weight("balanced", y_val)
    precision, recall_curve, _ = precision_recall_curve(y_val, y_pred_proba, sample_weight=w)
    no_skill = np.average(y_val, weights=w)

    plt.figure(figsize=(5, 4))
    plt.step(recall_curve, precision, where='post', label=f'Weighted PR (AP={pr_auc:.3f})')
    plt.hlines(no_skill, 0, 1, color='red', linestyle='--', label='No-skill')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()


from sklearn.metrics import recall_score, f1_score, average_precision_score
import pandas as pd

from sklearn.metrics import recall_score, f1_score, average_precision_score, confusion_matrix
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from sklearn.metrics import recall_score, f1_score, average_precision_score, confusion_matrix
import pandas as pd

def _evaluate_single_model(estimator, params, X_train, y_train, X_val, y_val):
    clean_params = {k: v for k, v in params.items() if pd.notna(v)}

    # Set parameters and train model
    model = estimator.set_params(**clean_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]  # Assumes binary classification

    recall = recall_score(y_val, y_pred, average='macro')
    f1     = f1_score(y_val, y_pred, average='macro')
    pr_auc = average_precision_score(y_val, y_proba)

    cm = confusion_matrix(y_val, y_pred)
    cm_str = f"[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]"

    return {
        **params,
        'recall_macro_val': recall,
        'f1_macro_val': f1,
        'pr_auc_val': pr_auc,
        'confusion_matrix': cm_str
    }

def evaluate_param_list(results_df, estimator, X_train, y_train, X_val, y_val, top_n=10, n_jobs=-3):

    # Automatically extract and clean parameter columns
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    param_dicts = (
        results_df.head(top_n)
        .rename(columns={col: col.replace('param_', '') for col in param_cols})
        .loc[:, [col.replace('param_', '') for col in param_cols]]
        .to_dict(orient='records')
    )

    # Parallel evaluation
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_single_model)(
            estimator, params, X_train, y_train, X_val, y_val
        )
        for params in param_dicts
    )

    return pd.DataFrame(results).sort_values("recall_macro_val", ascending=False)




