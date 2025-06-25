
# evaluator.py
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve
) 
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, f1_score, classification_report,
    average_precision_score, precision_recall_curve,
    confusion_matrix
)


class ModelEvaluator:
    def __init__(self, models: dict):
        self.models = models
        self.results = {}

    def _get_probabilities(self, clf, X):
        return clf.predict_proba(X)[:, 1]

    def evaluate(self, X_val, y_val, ylim=0.05, plot_cm=False, dpi=150):
        fig_pr, ax_pr = plt.subplots(figsize=(4.5, 3), dpi=dpi)
        plotted_any   = False
        cm_list       = []

        for name, clf in self.models.items():
            print(f" Evaluating {name}")
            try:
                y_pred  = clf.predict(X_val)
                y_proba = self._get_probabilities(clf, X_val)
            except Exception as e:
                print(f"   Skipped {name}: {e}")
                continue

            rec = recall_score(y_val, y_pred, average="macro")
            f1  = f1_score(y_val, y_pred, average="macro")
            ap  = average_precision_score(y_val, y_proba)
            self.results[name] = dict(recall_macro=rec, f1_macro=f1, pr_auc=ap)

            print(f"   Recall={rec:.4f} | F1={f1:.4f} | PR-AUC={ap:.4f}")
            print(classification_report(y_val, y_pred, digits=3))

            if plot_cm:
                cm_list.append((name, confusion_matrix(y_val, y_pred)))

            prec, rec_curve, _ = precision_recall_curve(y_val, y_proba)
            ax_pr.step(rec_curve, prec, where="post", lw=2, label=f"{name}  (AP={ap:.3f})")
            plotted_any = True

        if plotted_any:
            base_rate = y_val.mean()
            ax_pr.hlines(base_rate, 0, 1, colors="red", ls="--", label=f"Random (p={base_rate:.3f})")
            ax_pr.set_xlim(0, 1)
            ax_pr.set_ylim(0, ylim)
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision–Recall curves – all models")
            ax_pr.grid(True, alpha=0.3)
            ax_pr.legend(loc="upper right", fontsize=7)
            fig_pr.tight_layout()
            plt.show()
        else:
            print(" No PR curves drawn – every model errored out.")

        if plot_cm and cm_list:
            n = len(cm_list)
            if n == 1:
                name, cm = cm_list[0]
                fig, ax = plt.subplots(figsize=(3, 3), dpi=dpi)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
                ax.set_title(name, fontsize=10)
                ax.set_xlabel("Pred")
                ax.set_ylabel("Actual")
                fig.tight_layout()
                plt.show()
            else:
                cols = min(3, n)
                rows = ceil(n / cols)
                fig_cm, axes = plt.subplots(rows, cols,
                                            figsize=(cols * 3.2, rows * 3.2),
                                            dpi=dpi)
                axes = np.atleast_2d(axes).ravel()

                for ax, (name, cm) in zip(axes, cm_list):
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                                xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
                    ax.set_title(name, fontsize=9)
                    ax.set_xlabel("Pred")
                    ax.set_ylabel("Actual")

                for ax in axes[len(cm_list):]:
                    ax.set_visible(False)

                fig_cm.suptitle("Confusion matrices", y=1.02, fontsize=12)
                fig_cm.tight_layout()
                plt.show()

        return pd.DataFrame(self.results).T.sort_values("recall_macro", ascending=False)



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




