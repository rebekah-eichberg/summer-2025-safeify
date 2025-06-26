# Safeify
### Team members: [Emelie Arvidsson](https://github.com/EmmiArwid), [Rebekah Eichberg](https://github.com/rebekah-eichberg), [Alex Margolis](https://github.com/almargo), [Betul Senay Aras](https://github.com/betsenara) 

This is the repository for our Erdos Institute Summer 2025 Data Science Bootcamp Project.


## Project Overview

The goal of Safeify is to predict bad, hazardous, and low-quality products using Amazon product data and U.S. Consumer Product Safety Commission (CPSC) reports. Our main challenge was to develop a robust labeling strategy for identifying such products. 

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
