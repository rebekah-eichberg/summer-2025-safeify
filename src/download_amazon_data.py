# Download Amazon 2018 toy & games review and metadata
import pandas as pd
import os
from useful_functions import download_if_not_exists,extract_gz_file



# Download reviews
URL_REVIEWS="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Toys_and_Games.json.gz"
FILENAME_REVIEWS="../Data/amazon_reviews.json.gz"
download_if_not_exists(URL_REVIEWS,FILENAME_REVIEWS)
if os.path.exists(FILENAME_REVIEWS):
    extract_gz_file(FILENAME_REVIEWS,delete_original=True)

# Download metadata
URL_METADATA="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Toys_and_Games.json.gz"
FILENAME_METADATA="../Data/amazon_meta.json.gz"
download_if_not_exists(URL_METADATA,FILENAME_METADATA)
if os.path.exists(FILENAME_METADATA):
    extract_gz_file(FILENAME_METADATA,delete_original=True)


if not os.path.exists("../Data/metadata_raw.pkl"):
    print ("Pickling amazon product metadata.")
    metadata_df_raw = pd.read_json("../Data/amazon_meta.json",lines=True)
    metadata_df_raw.to_pickle('../Data/metadata_raw.pkl')
    del metadata_df_raw

if not os.path.exists("../Data/reviews_raw.pkl"):
    print ("Pickling amazon reviews.")
    reviews_df_raw = pd.read_json("../Data/amazon_reviews.json",lines=True)
    reviews_df_raw.to_pickle('../Data/reviews_raw.pkl')
    del metadata_df_raw
