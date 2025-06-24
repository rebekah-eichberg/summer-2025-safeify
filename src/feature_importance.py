

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel

def plot_top_features(model, X_train, y_train, top_n=20, title=None):
    model.fit(X_train, y_train)
    selector = SelectFromModel(model, prefit=True)
    mask = selector.get_support()
    selected_feats = X_train.columns[mask]
    print(f"Selected {len(selected_feats)} features out of {X_train.shape[1]}")

    # Get importance scores
    if hasattr(model, "coef_"):
        scores = model.coef_[0]
        score_type = "Coefficient Value"
        score_df = pd.DataFrame({
            'feature': X_train.columns,
            'score': scores,
            'abs_score': np.abs(scores)
        }).query("abs_score > 0").nlargest(top_n, 'abs_score')
        values = score_df['score'][::-1]
    elif hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
        score_type = "Importance"
        score_df = pd.DataFrame({
            'feature': X_train.columns,
            'score': scores
        }).query("score > 0").nlargest(top_n, 'score')
        values = score_df['score'][::-1]
    else:
        raise ValueError("Model must have either coef_ or feature_importances_ attribute.")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(score_df['feature'][::-1], values)
    plt.title(title or f"Top {top_n} Features")
    plt.xlabel(score_type)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return selected_feats.tolist()

