import os
from useful_functions import download_if_not_exists,verify_md5_file

FILENAMES=["test_v2.parquet",
           "validation_v2.parquet",
           "train_final_v2.parquet"]
URLS=["https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EWW1so88WM1Dqg9Oyc_KHogBPWW0AKSWUixuPDiPt3aanQ?e=82rPB3&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EUDRaTXlaTNGuQyZW00Rz6IB4ZeTOVZxPmYiefHpWPljNA?e=r5iCQk&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EVjFcmK5aChFgVcKLzClBxQBfAR0qrECYNDyCiucRrhR5Q?e=76A0wg&download=1"]



for i in range(len(FILENAMES)):
    download_if_not_exists(URLS[i],FILENAMES[i])


verify_md5_file("md5_checksums.txt")

