import os
from useful_functions import download_if_not_exists,verify_md5_file

FILENAMES=["test_v3.parquet",
           "validationA_v3.parquet",
           "validationB_v3.parquet",
           "train_final_v3.parquet"]
URLS=["https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EZI1QopV7zhHkWq0h47snu8B8mopSxIO4QHL0kpm4ftVGQ?e=g0RJQ3&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EYOhrHK0ej9JolFYomLla8YBrgfA3n1SRLh65pWveC_Iyw?e=kXs2dI&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/Ea5aQvHxqVNMqRlSIQjs_vkB5l0qId92Pis1lsApeT8bBw?e=M0komQ&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/ETk6h67B5cBLmU1ZWoaCergBobuFvTP1Vvp8ta38psi3qQ?e=PXJ4V3&download=1"]



for i in range(len(FILENAMES)):
    download_if_not_exists(URLS[i],FILENAMES[i])


verify_md5_file("md5_checksums.txt")



