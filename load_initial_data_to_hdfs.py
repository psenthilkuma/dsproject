import os
import subprocess

def upload_climate_data_to_hdfs(local_dir, hdfs_dir, target_files):
    """
    Uploads specified climate data files to HDFS in the required format.
    Files are renamed as <file_name>-<year>/<file_name>.
    
    Parameters:
        local_dir (str): Path to the local directory containing year-based subfolders.
        hdfs_dir (str): Base HDFS directory to upload files.
        target_files (list): List of specific filenames to upload (e.g., ['72494523293.csv']).
    """
    for year in os.listdir(local_dir):
        year_path = os.path.join(local_dir, year)
        if os.path.isdir(year_path):
            for file_name in os.listdir(year_path):
                if file_name in target_files:
                    local_file_path = os.path.join(year_path, file_name)
                    # Correcting the HDFS directory structure
                    hdfs_file_dir = f"{hdfs_dir}/{file_name.split('.')[0]}-{year}"
                    hdfs_file_path = f"{hdfs_file_dir}/{file_name}"

                    # Create the HDFS directory and upload the file
                    subprocess.run(f"hdfs dfs -mkdir -p {hdfs_file_dir}", shell=True, check=True)
                    subprocess.run(f"hdfs dfs -put -f {local_file_path} {hdfs_file_path}", shell=True, check=True)
                    print(f"Uploaded {local_file_path} to {hdfs_file_path}")


def upload_dengue_data_to_hdfs(local_file, hdfs_dir):
    """
    Uploads the Dengue data file to HDFS under /climate/raw/denguedata.
    
    Parameters:
        local_file (str): Path to the local Dengue data file.
        hdfs_dir (str): HDFS directory to upload the file.
    """
    hdfs_file_path = f"{hdfs_dir}/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv"

    # Create the HDFS directory and upload the file
    subprocess.run(f"hdfs dfs -mkdir -p {hdfs_dir}", shell=True, check=True)
    subprocess.run(f"hdfs dfs -put -f {local_file} {hdfs_file_path}", shell=True, check=True)
    print(f"Uploaded {local_file} to {hdfs_file_path}")


if __name__ == "__main__":
    # Local directories
    climate_data_dir = "/Users/spattusa/Downloads/DS/Climate"
    dengue_data_file = "/Users/spattusa/Downloads/DS/DengueData/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv"

    # HDFS directories
    climate_hdfs_dir = "/climate/raw"
    dengue_hdfs_dir = "/climate/raw/denguedata"

    # Configurable list of target filenames
    target_filenames = ["72494523293.csv"]

    # Upload Climate Data
    upload_climate_data_to_hdfs(climate_data_dir, climate_hdfs_dir, target_filenames)

    # Upload Dengue Data
    upload_dengue_data_to_hdfs(dengue_data_file, dengue_hdfs_dir)