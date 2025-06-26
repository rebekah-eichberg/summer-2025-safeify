# Modeling Folder

This folder contains all Jupyter notebooks and scripts related to machine learning model development, evaluation, and analysis for the Safeify project.

## Contents

- **MLModelTrainingHyperparamterTuning.ipynb**  
  Main notebook for training machine learning models, performing hyperparameter tuning, and evaluating model performance using cross-validation. Includes feature selection, model comparison, and metric reporting.

- **MLModelTesting.ipynb**  
  Notebook for testing trained models on the test set. Includes performance metrics, error analysis, and final model selection.

- **prob_cal.ipynb**  
  Notebook for probability calibration of model outputs. Contains code for calibrating predicted probabilities, plotting reliability curves, and analyzing probability distributions. It also contains the code for anomaly detections with distribution plots.

- **shap_feature_selection.ipynb**  
  Notebook for feature importance analysis and selection using SHAP (SHapley Additive exPlanations). Helps identify the most and least important features for model performance.

- **false_positive_analysis.ipynb**  
  In-depth analysis of false positives and false negatives. Includes category concentration, review text inspection, and error breakdowns to understand model mistakes.

## Notes

- All notebooks expect the processed data splits and feature files from the `Data` folder. These files can be downloaded by running the ``../src/download_split_finaldata.py``.
- Run the notebooks in the order listed above.


