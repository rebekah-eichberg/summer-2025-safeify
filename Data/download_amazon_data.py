# Download Amazon 2018 toy & games review and metadata
import os
from useful_functions import download_if_not_exists,extract_gz_file

URL_REVIEWS="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Toys_and_Games.json.gz"
FILENAME_REVIEWS="amazon_reviews.json.gz"
download_if_not_exists(URL_REVIEWS,FILENAME_REVIEWS)
if os.path.exists(FILENAME_REVIEWS):
    extract_gz_file(FILENAME_REVIEWS,delete_original=True)


URL_METADATA="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Toys_and_Games.json.gz"
FILENAME_METADATA="amazon_meta.json.gz"
download_if_not_exists(URL_METADATA,FILENAME_METADATA)
if os.path.exists(FILENAME_METADATA):
    extract_gz_file(FILENAME_METADATA,delete_original=True)
