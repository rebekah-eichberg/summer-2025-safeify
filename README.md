# Safeify
### Team members: [Emelie Arvidsson](https://github.com/EmmiArwid), [Rebekah Eichberg](https://github.com/rebekah-eichberg), [Alex Margolis](https://github.com/almargo), [Betul Senay Aras](https://github.com/betsenara) 

This is the repository our Erdos Institute Summer 2025 Data Science Bootcamp Project.

## Overview of the project
The goal of the project is to predict bad, hazardous, and low-quality products, leaving the task of identifying best-sellers to future work. A main part of the
project has been to identify a way to label products as bad, low-quality, or hazardous. We addressed this challenge by leveraging the U.S. Consumer
Product Safety Commission (CPSC) database, using both recall and complaint reports to identify harmful products.

## Structure of the repository

### Modelling

### Data

### [Matching Amazon.com products with incident reports](matching/)
The ``matching`` folder contains notebooks and scripts used to match for matching Amazon product metadata to incident and recall reports for toys and children’s products. The matching process combines fuzzy string matching, semantic similarity using transformer models, web search enrichment, and large language model (LLM) verification.
More information is contained in the [readme file](matching/) of this folder.

### [Cleaning and feature engineering](cleaning_and_feature_engineering/)
The `cleaning_and_feature_engineering` directory contains notebooks and scripts for cleaning Amazon product metadata and review data, engineering features, and builds the main dataset for modeling. Please see the [readme file](cleaning_and_feature_engineering/) for more information.


## Requirements
Running 
```
pip install -r requirements.txt
```
will install all the required python modules.
