from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import os

default_args = {
    "owner": "file_detection_pipeline",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Function to detect files in the monitored folder
def detect_files_in_folder(folder_path, **kwargs):
    files = []
    for root, _, filenames in os.walk(folder_path):  # Traverse subdirectories
        for filename in filenames:
            files.append(os.path.join(root, filename))
    # Push the list of files to XCom
    ti = kwargs["ti"]
    ti.xcom_push(key="detected_files", value=files)
    print(f"Detected files: {files}")

with DAG(
    "dsproject_file_detection_dag",
    default_args=default_args,
    description="Detect files and trigger the processing DAG",
    schedule_interval=None,
    start_date=datetime(2024, 12, 1),
    catchup=False,
) as dag:

    # Monitor folder for new data
    monitor_folder = FileSensor(
        task_id="monitor_folder",
        filepath="/Users/spattusa/Downloads/DS/Monitor",
        poke_interval=10,
        timeout=600,
    )

    # Detect files in folder
    detect_files = PythonOperator(
        task_id="detect_files",
        python_callable=detect_files_in_folder,
        op_args=["/Users/spattusa/Downloads/DS/Monitor"],
        provide_context=True,
    )

    # Trigger the second DAG for each detected file
    def trigger_processing_dag_per_file(**kwargs):
        ti = kwargs["ti"]
        files = ti.xcom_pull(task_ids="detect_files", key="detected_files")
        if not files:
            raise ValueError("No files detected. Please check the Monitor folder.")
        for file_path in files:
            TriggerDagRunOperator(
                task_id=f"trigger_dag_for_{os.path.basename(file_path)}",
                trigger_dag_id="dsproject_file_processing_dag",  # The ID of the second DAG
                conf={"file_path": file_path},  # Pass the file path as input
            ).execute(context=kwargs)
            print(f"Triggered processing for file: {file_path}")

    trigger_processing_tasks = PythonOperator(
        task_id="trigger_processing_tasks",
        python_callable=trigger_processing_dag_per_file,
        provide_context=True,
    )

    # DAG dependencies
    monitor_folder >> detect_files >> trigger_processing_tasks