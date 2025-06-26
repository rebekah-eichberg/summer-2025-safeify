# Safeify /src Folder

This folder contains all scripts and helper modules for data downloading, preprocessing, feature engineering, splitting, and evaluation for the Safeify project.

---

## Scripts

- **download_amazon_data.py**  
  Downloads raw Amazon product metadata and reviews, and converts them to pickle format.

- **download_embeddings.py**  
  Downloads precomputed text and summary embeddings for reviews.

- **download_split_finaldata.py**  
  Downloads the final train/validation/test splits and cross-validation indices for modeling.

- **features.py**  
  Functions for feature transformation pipelines.

- **feature_importance.py**  
  Utilities for analyzing and plotting feature importance from trained models.

- **custom_ttsplit.py**  
  Implements stratified group splits for train/test/validation, preserving label and component integrity.

- **get_cv_split.py**  
  Loads and manages cross-validation splits using predefined indices.

- **generate_component_nums.py**  
  Assigns connected component numbers to products based on matches.

- **find_datafiles.py**  
  Scans the project for data file usage (reads/writes) in code and notebooks.

- **helper_functions.py**  
  Text processing, cleaning, and zero-shot classification helpers.

- **prob_cal_helper_functions.py**  
  Functions for probability calibration, plotting reliability curves, and splitting features/targets.

- **evaluator.py**  
  Model evaluation utilities, including precision-recall curves and confusion matrices.

- **scrape_duckduckgo.py**  
  Utilities for scraping DuckDuckGo and extracting Amazon ASINs from search results.

- **useful_functions.py**  
  General-purpose utilities for downloading files, extracting archives, and verifying MD5 checksums.

For more details on each script, see the docstrings in the respective `.py` files.
