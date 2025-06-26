import os
from useful_functions import download_if_not_exists,verify_md5_file

FILENAMES=["../Data/agg_summary_embeddings.pkl",
           "../Data/reviewtext_features_df.pkl"]
URLS=[
    "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EbkxqErVR05LqpZLDDeV-GUBx6o6QPPNFn4zTP8kP0s89Q?e=B0TCEI&download=1",
    "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/Ecds8B2Rl0lErYH2XiZQl30BqgW3AB-ZuyoeXzdBn99ecQ?e=S5gsh4&download=1"
    ]

for i in range(len(FILENAMES)):
    download_if_not_exists(URLS[i],FILENAMES[i])
