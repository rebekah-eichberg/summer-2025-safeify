import os
from useful_functions import download_if_not_exists,verify_md5_file


URL_TEST="https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EW-2lTPvnYJEjlI5pjlICCMB-Y3YOok_u8ZhOE2EC5esNQ?e=sbS8EX&download=1"
FILENAME_TEST="test_v1.parquet"
URL_TRAIN="https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EcjEJGg0mUVAuLHIzzpDLRgBL5xj38zEurzpc_2dgr_g9A?e=CwyvCM&download=1"
FILENAME_TRAIN="train_v1.parquet"


download_if_not_exists(URL_TEST,FILENAME_TEST)
download_if_not_exists(URL_TRAIN,FILENAME_TRAIN)


verify_md5_file("md5_checksums.txt")

