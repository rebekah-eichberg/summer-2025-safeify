import os
from useful_functions import download_if_not_exists,verify_md5_file

FILENAMES=["test_v1.parquet",
           "train_v1.parquet",
           "validation_v1.parquet",
           "train_final_v1.parquet"]
URLS=["https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EW-2lTPvnYJEjlI5pjlICCMB-Y3YOok_u8ZhOE2EC5esNQ?e=sbS8EX&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EcjEJGg0mUVAuLHIzzpDLRgBL5xj38zEurzpc_2dgr_g9A?e=CwyvCM&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EX9g9XsEwFhHvdx-moe2SJwByp6CLqavq8GhuSty33uenQ?e=wR4vEy&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EXSUALn38fBNmM-BSgp4GBMB5PDhFDsdQMAXs1z2JeNBIQ?e=fwdMZq&download=1"]



for i in range(len(FILENAMES)):
    download_if_not_exists(URLS[i],FILENAMES[i])


verify_md5_file("md5_checksums.txt")

