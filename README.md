# Safeify
### Team members: [Emelie Arvidsson](https://github.com/EmmiArwid), [Rebekah Eichberg](https://github.com/rebekah-eichberg), [Alex Margolis](https://github.com/almargo), [Betul Senay Aras](https://github.com/betsenara) 

This is the repository for our Erdos Institute Summer 2025 Data Science Bootcamp Project.


## Project Overview

The goal of Safeify is to predict bad, hazardous, and low-quality products using Amazon product data and U.S. Consumer Product Safety Commission (CPSC) reports. Our main challenge was to develop a robust labeling strategy for identifying such products. 

## Stakeholders and Key Performance Indicators (KPIs)
Safeify is a model that is appealing to the following parties: 
- U.S. Consumer Product Safety Commission (CPSC) to incorporate recall and incident reports in real time with e-commerce platforms to bring information to consumers at a quicker rate
- Small online retailers on large e-commerce platforms such as Amazon trying to understand true performance/likeability of their product
- E-commerce platforms looking to lean out their inventory by reducing sales of low quality and problematic products

The KPIs are:
- Predictive: Correctly identifying products that are low quality and unsafe; ability to investigate flagged items
Safety Impact: Reduction in time to flag safety issues 
- Consumer Satisfaction: Decrease in negative reviews or returns or movement towards more credible reviews; Consumer trust and engagement with the Safeify model
- Retailer: Reduction in inventory holding of low quality and hazardous, decrease in customer support tickets related to flagged products

## Data Sources:
- Amazon Reviews and Metadata for product information such as reviews and rating (https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- CPSC Recalls and Incident Reports to create labels for products that appear on Amazon that are unsafe and low quality (https://www.saferproducts.gov/PublicSearch)

## Methods, Modelling and Results:
We trained a variation of classifiers including Logistic Regression, Random Forest and XGBoost. To train the models, we developed a custom train/test split that used a graph and split the data basde on connected components to prevent data leakage in any of our sets. This was a bit challenging because as we continued to split the data, we had even smaller sets with less class 1 observations. We later learned through exploration of feature selection that the feature 'category' was unevenly distributed across the sets which led to ambiguous results. 

We used hyperparameter tuning to adjust the weights to manage the severely imbalanced data. This led us to exploring variations of models that either had balanced weighting or more aggressive weighting. We did the tuning with the models aforementioned and also used Voting and Stacking Classifiers. Trying to minimize false positives and false negatives and maximize macro average recall, we continued to end up in three regimes that involved trade offs of all those metrics. Thus, it is up to the stakeholder and business to determine what they prioritize.

Ultimately, we found that our final (Logistic Regression, Random Forest, and Voting Classifier of the three base estimators) generalized well and were not overfitting. The results we saw were consistent and slightly better in average precision score than on our validation sets.

Lastly, we used probability calibration to calibrate the model to predict true likelihoods to perform anomaly detection. When using the Voting Classifer and looking at the 90th percentile, the model is able to detect 4000 anomalies, meaning, true class 0 products with predicted probability of class 1 higher than average. These anomalies are flagged by the model and can be manually reviewed.

## Repository Structure

- **[modeling/](modeling/)**  
  Contains Jupyter notebooks and scripts for machine learning model development, feature selection, training, evaluation, probability calibration, and error analysis. See the [modeling README](modeling/ReadMe.md) for details.

- **[Data/](Data/)**  
  Contains all raw, intermediate, and processed datasets, as well as scripts for downloading and verifying data. See the [Data README](Data/README.md) for a full list and description of files.

- **[matching/](matching/)**  
  Notebooks and scripts for matching Amazon product metadata to CPSC incident and recall reports for toys and children’s products. The matching process uses fuzzy string matching, transformer-based semantic similarity, web searching, and LLM verification. More information is in the [matching README](matching/README.md).

- **[cleaning_and_feature_engineering/](cleaning_and_feature_engineering/)**  
  Notebooks and scripts for cleaning Amazon product metadata and review data, engineering features, and building the main dataset for modeling. See the [cleaning_and_feature_engineering README](cleaning_and_feature_engineering/README.md) for more information.

- **[src/](src/)**  
  Python scripts for data downloading, preprocessing, feature engineering, splitting, and utility functions.

## Getting Started

1. **Install requirements:**  
   ```
   pip install -r requirements.txt
   ```

2. **Download data:**  
   Use the scripts in the `src/` folder to download raw data, embeddings, and processed splits as described in the [Data README](Data/README.md).

3. **Explore and run notebooks:**  
   See the individual folder READMEs for recommended notebook order and workflow.
