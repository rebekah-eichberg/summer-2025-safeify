import os
from useful_functions import download_if_not_exists,verify_md5_file

# Get the absolute path to the directory the script is in
script_dir = os.path.dirname(os.path.abspath(__file__))

# Then use script_dir to build paths
data_path = os.chdir(script_dir)

FILENAMES=["../Data/test_v3.parquet",
           "../Data/validationA_v3.parquet",
           "../Data/validationB_v3.parquet",
           "../Data/train_final_v3.parquet",
           "../Data/CV_val_split.parquet"]
URLS=["https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EZI1QopV7zhHkWq0h47snu8B8mopSxIO4QHL0kpm4ftVGQ?e=g0RJQ3&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EYOhrHK0ej9JolFYomLla8YBrgfA3n1SRLh65pWveC_Iyw?e=kXs2dI&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/Ea5aQvHxqVNMqRlSIQjs_vkB5l0qId92Pis1lsApeT8bBw?e=M0komQ&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/ETk6h67B5cBLmU1ZWoaCergBobuFvTP1Vvp8ta38psi3qQ?e=PXJ4V3&download=1",
      "https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/margolis_93_osu_edu/EY6CafSML3lFg8M72u2nMccBTxA5vOrj5pXeM2TZro8CDw?e=KJeOn7&download=1"]



for i in range(len(FILENAMES)):
    download_if_not_exists(URLS[i],FILENAMES[i])

os.chdir("../Data/")
verify_md5_file("md5_checksums.txt")



