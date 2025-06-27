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

## Project Summary

We worked with two datasets:  
- **600K+ unlabeled Amazon metadata** (including product reviews and attributes)  
- **CPSC complaints data** (product recall reports)

We used **fuzzy matching** and **ASIN scraping** to match Amazon products with recall data and generate labels. This process yielded a **very small number of positive class (label 1) samples (~1,500)** and a vast number of negative class (label 0) samples, resulting in an **extremely imbalanced dataset**. Additionally, due to the similarity among Amazon products, many-to-many relationships formed between matched products and recalls.

To **prevent data leakage**, especially among similar products, we developed a **custom train/test split** strategy using a **graph-based approach**. We built connected components based on product matches and ensured that no component was split across training, validation, or test sets.

---

## Feature Engineering

We generated a variety of features from both **metadata** and **reviews**:

- **From metadata:**
  - Product rank
  - Price
  - Category

- **From reviews:**
  - Statistical features (e.g., number of reviews, average rating)
  - Bot indicators (e.g., repeated reviewers)
  - Embeddings and similarity scores from review text and summary
  - Sentiment scores to reduce false positives with positive-sounding reviews

---

## Modeling & Imbalance Handling

To reduce class imbalance, we **undersampled** the negative class to 200K samples.  
We trained several classifiers including:

- Logistic Regression  
- Random Forest  
- XGBoost  
- Voting Classifier (ensemble of the above)

We applied **hyperparameter tuning**, adjusting **class weights** to explore both **balanced** and **aggressively weighted** regimes.

Our goal was to:
- **Maximize macro average recall**
- **Minimize false positives and false negatives**

This led us to three modeling regimes with different trade-offs. We leave it to the **stakeholders** to choose a strategy that aligns best with business goals.

---

## Feature Impact & Data Splitting Challenges

During feature analysis, we noticed many false positives had negative reviews. When comparing **false positives vs. true negatives**, we found **different category distributions**, which led us to investigate the role of the `'category'` feature.

We experimented with including/excluding the category feature, but results were **inconsistent** between validation and cross-validation. We realized this was due to our **custom component-based data split**, which assigned similar products to the same set — resulting in a **distribution mismatch** of the `'category'` feature between training and validation sets.

Despite these challenges, our models achieved high recall (~**0.8**) and were able to capture most true positives — although at the cost of higher false positives in some configurations.

---

## Probability Calibration & Anomaly Detection

We applied **probability calibration** to ensure the model's predicted probabilities reflected **true likelihoods**, allowing for **anomaly detection** use cases.

Using the calibrated **Voting Classifier**:
- At the **90th percentile**, the model flagged ~**4,000 potential anomalies** (i.e., class 0 products with high predicted probability of being class 1)
- These high-risk products could be **manually reviewed** for further action

---

## Final Model Evaluation on Test Set

We evaluated our **uncalibrated final models** on the held-out test set and found:

- No overfitting
- Performance was **stable and slightly improved** compared to validation results
- Macro recall and average precision remained consistent

### Test Set Performance Summary

| Model                | Macro Recall | Macro F1  | PR AUC    |
|----------------------|--------------|-----------|-----------|
| Voting Classifier    | **0.7820**   | 0.4412    | 0.1150    |
| Logistic Regression  | 0.7713       | 0.4409    | 0.0974    |
| XGBoost              | 0.7653       | 0.4388    | 0.0980    |
| Random Forest        | 0.7599       | 0.4205    | **0.1184** |


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
