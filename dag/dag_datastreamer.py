from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import shutil
from kafka import KafkaProducer

default_args = {
    "owner": "file_processing_pipeline",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Function to upload files to HDFS
def upload_to_hdfs(file_path, hdfs_dir, **kwargs):
    # Extract the directory and file name
    directory, base_name = os.path.split(file_path)
    
    # Extract the year from the directory path
    year = os.path.basename(directory)
    
    # Create the new HDFS directory structure
    new_hdfs_dir = os.path.join(hdfs_dir, f"{base_name.split('.')[0]}-{year}")
    
    # Ensure the HDFS directory exists
    os.system(f"hdfs dfs -mkdir -p {new_hdfs_dir}")
    
    # Upload the file to the new HDFS path
    os.system(f"hdfs dfs -put {file_path} {new_hdfs_dir}")
    
    print(f"Uploaded {file_path} to HDFS as {new_hdfs_dir}/{base_name}")

# Function to stream data from HDFS to Kafka
def stream_to_kafka(file_path, kafka_topic, **kwargs):
    """
    Streams data from a local file to a Kafka topic.

    Parameters:
        file_path (str): The path of the local file.
        kafka_topic (str): The Kafka topic to publish the data to.
    """
    producer = KafkaProducer(bootstrap_servers="localhost:9092")

    try:
        # Read the file content directly
        with open(file_path, "r") as file:
            for line in file:
                producer.send(kafka_topic, line.encode("utf-8"))
                print(f"Sent line to Kafka: {line.strip()}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Failed to stream file {file_path}: {e}")
    finally:
        producer.close()
        print("Data streaming to Kafka completed.")

# Function to archive files
def archive_file(file_path, archive_folder, **kwargs):
    os.makedirs(archive_folder, exist_ok=True)
    file_name = os.path.basename(file_path)
    shutil.move(file_path, os.path.join(archive_folder, file_name))
    print(f"Archived {file_path} to {archive_folder}")

with DAG(
    "dsproject_file_processing_dag",
    default_args=default_args,
    description="Process individual files",
    schedule_interval=None,
    start_date=datetime(2024, 12, 1),
    catchup=False,
) as dag:

    # Task 1: Upload file to HDFS
    upload_to_hdfs_task = PythonOperator(
        task_id="upload_to_hdfs",
        python_callable=upload_to_hdfs,
        op_args=["{{ dag_run.conf['file_path'] }}", "/climate/raw"],
    )

    # Task 2: Stream file to Kafka
    stream_to_kafka_task = PythonOperator(
        task_id="stream_to_kafka",
        python_callable=stream_to_kafka,
        op_args=["{{ dag_run.conf['file_path'] }}", "climate-sample"],
    )

    # Task 3: Archive file
    archive_file_task = PythonOperator(
        task_id="archive_file",
        python_callable=archive_file,
        op_args=["{{ dag_run.conf['file_path'] }}", "/Users/spattusa/Downloads/DS/Archive"],
    )

    # DAG dependencies
    upload_to_hdfs_task >> stream_to_kafka_task >> archive_file_task